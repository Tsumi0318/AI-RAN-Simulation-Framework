#!/usr/bin/env python3
"""Mark4: PDF-aligned, data-driven AI-RAN offloading experiment.

The experiment keeps the PDF's asynchronous best-response game and required
baselines/metrics, while making heterogeneous task memory mathematically
consistent through a marginal barrier cost. DeepSeek is used as an independent
semantic coordinator. Formula validation, not free-form LLM arithmetic, drives
the strategy update.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "00_原始数据" / "GenTD26"
OUT = ROOT / "02_表格数据"


@dataclass(frozen=True)
class Config:
    seed: int = 20260717
    n_main: int = 100
    n_pool: int = 200
    peak_hour: str = "2024-12-06 00:00:00"
    t_max: int = 1500
    no_change_stop: int = 100
    random_trials: int = 500
    sweep_trials: int = 30
    sweep_n_min: int = 30
    sweep_n_max: int = 200
    sweep_n_step: int = 10
    vram_capacity_gb_assumed: float = 16.0
    equivalent_capacity_tasks_assumed: int = 80
    memory_pressure_at_capacity_cost_assumed: float = 5.0
    edge_power_w_assumed: float = 10.0
    local_slowdown: float = 1.8
    local_energy_addition: float = 0.4
    tx_base_cost: float = 0.15
    prompt_tx_weight: float = 0.15
    radio_energy_mj_per_kb_assumed: float = 0.2
    request_metadata_kb_assumed: float = 1.0
    utf8_bytes_per_char_assumed: float = 2.0
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 90.0
    llm_retries: int = 3
    semantic_api_workers: int = 4


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest() -> list[dict[str, Any]]:
    source = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI"
    rows = []
    for path in sorted(DATA.glob("*")):
        if path.is_file():
            rows.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source": source,
                "status": "official_anonymized_production_trace",
            })
    return rows


def load_request_pool(c: Config) -> pd.DataFrame:
    req = pd.read_csv(DATA / "lora_request_trace.csv")
    req["gmt_create"] = pd.to_datetime(req["gmt_create"])
    start = pd.Timestamp(c.peak_hour)
    usable = req.loc[
        (req.predict_status == "SUCCEED")
        & (req.gmt_create >= start)
        & (req.gmt_create < start + pd.Timedelta(hours=1))
        & (req.exec_time_seconds > 0)
    ].copy()
    if len(usable) < c.n_pool:
        raise ValueError(f"Peak hour has {len(usable)} usable requests; need {c.n_pool}")
    return (
        usable.sample(c.n_pool, random_state=c.seed)
        .sort_values("gmt_create")
        .reset_index(drop=True)
    )


def fit_mm1_delay(c: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    queue_size = (
        pd.read_csv(DATA / "queue_size_raw_anon.csv")
        .groupby("timestamp_anon").value.median().rename("queue_size")
    )
    queue_rt = (
        pd.read_csv(DATA / "queue_rt_raw_anon.csv")
        .groupby("timestamp_anon").value.median().rename("queue_delay_ms")
    )
    points = pd.concat([queue_size, queue_rt], axis=1, join="inner").dropna().reset_index()
    queue_reference = max(float(points.queue_size.quantile(0.99)), 1e-9)
    points["k_equivalent"] = np.clip(
        points.queue_size / queue_reference * c.equivalent_capacity_tasks_assumed,
        0,
        1.25 * c.equivalent_capacity_tasks_assumed,
    )
    x = points.k_equivalent.to_numpy(float)
    y = np.maximum(points.queue_delay_ms.to_numpy(float) / 1000.0, 1e-6)

    # D(K)=d0/(1-rho*K/Kcap) is exactly 1/(mu-lambda*K).
    def residual(z: np.ndarray) -> np.ndarray:
        d0, rho = z
        pred = d0 / np.maximum(1.0 - rho * x / c.equivalent_capacity_tasks_assumed, 1e-6)
        return np.log(pred) - np.log(y)

    result = least_squares(
        residual,
        x0=np.array([float(np.median(y)), 0.2]),
        bounds=([1e-4, 0.0], [10.0, 0.95]),
        loss="soft_l1",
    )
    d0, rho = map(float, result.x)
    mu = 1.0 / d0
    lam = mu * rho / c.equivalent_capacity_tasks_assumed
    pred = 1.0 / np.maximum(mu - lam * x, 1e-9)
    points["queue_fit_ms"] = pred * 1000.0
    return {
        "formula": "D_comp(K)=1/(mu-lambda*K)",
        "mu_per_second": mu,
        "lambda_per_task_per_second": lam,
        "rho_at_assumed_capacity": rho,
        "stability_limit_k": mu / lam if lam > 0 else None,
        "queue_reference_p99": queue_reference,
        "equivalent_capacity_tasks_assumed": c.equivalent_capacity_tasks_assumed,
        "matched_timestamp_points": len(points),
        "r2_seconds": float(r2_score(y, pred)),
        "fit_method": "robust log-residual fit on exact timestamp matches of gateway queue size and queue delay",
        "caveat": "Queue size is anonymized/scaled and its observed association with delay is weak; fitted mu/lambda are simulation proxies, not identified physical rates.",
    }, points


def fit_vram_barrier(c: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = pd.read_csv(DATA / "pod_gpu_memory_used_bytes_anon.csv")
    values = raw.loc[raw.value > 0, "value"].to_numpy(float)
    vmax_proxy = float(np.quantile(values, 0.99))
    quantiles = np.linspace(0.05, 0.995, 80)
    utilization = np.quantile(values / vmax_proxy, quantiles)
    # Empirical upper-tail shape. Its conversion to normalized game cost is an
    # explicit scale assumption because the source trace has no cost units.
    target = (
        (1.0 / np.maximum(1.0 - quantiles, 1e-6))
        / 100.0
        * c.memory_pressure_at_capacity_cost_assumed
    )

    def barrier(u: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha * np.exp(np.clip(beta * (u - 1.0), -30, 30))

    params, _ = curve_fit(
        barrier,
        utilization,
        target,
        p0=(1.0, 8.0),
        bounds=([1e-6, 0.0], [1e4, 50.0]),
        maxfev=100000,
    )
    pred = barrier(utilization, *params)
    points = pd.DataFrame({
        "quantile": quantiles,
        "vram_utilization_proxy": utilization,
        "empirical_tail_pressure": target,
        "barrier_fit": pred,
    })
    return {
        "pdf_formula": "M(K)=alpha*exp(beta*(K*v_req-Vmax))",
        "implemented_normalized_formula": "B(V)=alpha*exp(beta*(V/Vmax-1))",
        "alpha": float(params[0]),
        "beta": float(params[1]),
        "vmax_data_proxy_bytes": vmax_proxy,
        "vmax_data_definition": "observed positive per-pod VRAM P99",
        "r2_tail_pressure": float(r2_score(target, pred)),
        "pressure_cost_at_capacity_assumed": c.memory_pressure_at_capacity_cost_assumed,
        "fit_method": "exponential fit to empirical upper-tail pressure normalized at the observed P99",
        "caveat": "Beta and the tail shape come from anonymized production VRAM fluctuations; the cost scale at capacity and the 16 GB simulator capacity are explicit assumptions.",
    }, points


def scheduler_summary() -> dict[str, float]:
    pipeline = pd.read_csv(DATA / "pipeline_update_latency_anon.csv").value.to_numpy(float)
    inference = pd.read_csv(DATA / "model_predict_data_anon.csv").value.to_numpy(float)
    pipeline = pipeline[pipeline > 0]
    inference = inference[inference > 0]
    return {
        "pipeline_update_median_ms": float(np.median(pipeline)),
        "pipeline_update_p95_ms": float(np.quantile(pipeline, 0.95)),
        "pure_inference_median_ms": float(np.median(inference)),
        "pure_inference_p95_ms": float(np.quantile(inference, 0.95)),
    }


def build_intents(pool: pd.DataFrame, c: Config) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    exec_s = pool.exec_time_seconds.to_numpy(float)
    exec_norm = exec_s / max(float(np.median(exec_s)), 1e-9)
    prompt_chars = pool.prompt_length.fillna(pool.prompt_length.median()).to_numpy(float)
    neg_chars = pool.negative_prompt_length.fillna(0).to_numpy(float)
    steps = pool.num_inference_steps.fillna(pool.num_inference_steps.median()).to_numpy(float)
    images = pool.num_images_per_prompt.fillna(1).to_numpy(float)
    lora_count = pool.num_lora.fillna(0).to_numpy(float)
    prompt_scale = max(float(np.quantile(prompt_chars, 0.95)), 1.0)

    e_loc_norm = c.local_slowdown * exec_norm + c.local_energy_addition
    e_tx_norm = c.tx_base_cost + c.prompt_tx_weight * np.minimum(prompt_chars / prompt_scale, 1.5)
    local_energy_mj = c.edge_power_w_assumed * c.local_slowdown * exec_s * 1000.0
    payload_kb = c.request_metadata_kb_assumed + (prompt_chars + neg_chars) * c.utf8_bytes_per_char_assumed / 1024.0
    tx_energy_mj = c.radio_energy_mj_per_kb_assumed * payload_kb

    intents: list[dict[str, Any]] = []
    for i, row in pool.iterrows():
        intents.append({
            "node": int(i),
            "predict_type": str(row.predict_type),
            "prompt_length_chars": int(round(prompt_chars[i])),
            "negative_prompt_length_chars": int(round(neg_chars[i])),
            "steps": int(round(steps[i])),
            "num_images": int(round(images[i])),
            "lora": bool(lora_count[i] > 0),
            "lora_count": int(round(lora_count[i])),
            "observed_exec_time_seconds": float(exec_s[i]),
            "local_energy_cost_normalized": float(e_loc_norm[i]),
            "tx_energy_cost_normalized": float(e_tx_norm[i]),
            "local_energy_mj_simulated": float(local_energy_mj[i]),
            "tx_energy_mj_simulated": float(tx_energy_mj[i]),
        })
    return intents, {
        "exec_s": exec_s,
        "exec_norm": exec_norm,
        "prompt_chars": prompt_chars,
        "steps": steps,
        "images": images,
        "lora_count": lora_count,
        "e_loc": e_loc_norm,
        "e_tx": e_tx_norm,
        "local_energy_mj": local_energy_mj,
        "tx_energy_mj": tx_energy_mj,
    }


class DeepSeekClient:
    def __init__(self, c: Config):
        self.c = c
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", c.deepseek_model)
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the full experiment")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=c.llm_timeout_seconds)

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def complete_json(self, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.c.llm_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                parsed = self.parse_json(response.choices[0].message.content or "")
                usage = response.usage
                return {
                    "parsed": parsed,
                    "resolved_model": str(response.model),
                    "latency_ms": latency_ms,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "attempts": attempt + 1,
                    "parse_ok": True,
                }
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
        raise RuntimeError(f"DeepSeek failed after retries: {last_error}")


class SemanticIntentParser:
    def __init__(self, client: DeepSeekClient, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.lock = threading.Lock()

    def evaluate(self, intent: dict[str, Any]) -> dict[str, Any]:
        key = hashlib.sha256(json.dumps(intent, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                result = dict(self.cache[key])
                result["cache_hit"] = True
                return result
        system = (
            "You are a deterministic resource-intent parser for an AI inference scheduler. "
            "Estimate task-relative compute and VRAM multipliers using only prompt length, diffusion steps, "
            "image count, task type, LoRA count, and observed execution time. Return JSON only with keys "
            "compute_multiplier, vram_multiplier, risk_level, semantic_warning. Multipliers must be numbers "
            "from 0.5 to 2.5. Do not decide local/offload."
        )
        response = self.client.complete_json(system, intent, max_tokens=180)
        raw = response.pop("parsed")
        risk = str(raw.get("risk_level", "medium")).lower()
        if risk not in {"low", "medium", "high"}:
            risk = "medium"
        result = {
            "node": intent["node"],
            "compute_multiplier": float(np.clip(safe_float(raw.get("compute_multiplier"), 1.0), 0.5, 2.5)),
            "vram_multiplier": float(np.clip(safe_float(raw.get("vram_multiplier"), 1.0), 0.5, 2.5)),
            "risk_level": risk,
            "semantic_warning": str(raw.get("semantic_warning", ""))[:300],
            "model": self.client.model,
            **response,
            "cache_hit": False,
        }
        with self.lock:
            self.cache[key] = result
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


class FormulaModel:
    def __init__(
        self,
        c: Config,
        arrays: dict[str, np.ndarray],
        semantics: pd.DataFrame,
        queue: dict[str, Any],
        memory: dict[str, Any],
        scheduler: dict[str, float],
        n: int,
    ):
        self.c = c
        self.n = n
        self.a = {key: value[:n].copy() for key, value in arrays.items()}
        self.compute_multiplier = semantics.compute_multiplier.to_numpy(float)[:n]
        self.vram_multiplier = semantics.vram_multiplier.to_numpy(float)[:n]
        cold_start_ratio = scheduler["pipeline_update_median_ms"] / max(scheduler["pure_inference_median_ms"], 1e-9)
        self.center_compute = self.a["exec_norm"] * self.compute_multiplier + cold_start_ratio * (self.a["lora_count"] > 0)
        base_vram_gb = c.vram_capacity_gb_assumed / c.equivalent_capacity_tasks_assumed
        self.vram_gb = base_vram_gb * self.vram_multiplier
        self.vram_fraction = self.vram_gb / c.vram_capacity_gb_assumed
        self.mu = float(queue["mu_per_second"])
        self.lam = float(queue["lambda_per_task_per_second"])
        self.alpha = float(memory["alpha"])
        self.beta = float(memory["beta"])

    def dcomp(self, k: int) -> float:
        if k <= 0:
            return 0.0
        denominator = self.mu - self.lam * k
        return float(1.0 / denominator) if denominator > 0 else 1e6

    def barrier(self, v_fraction: float) -> float:
        return float(self.alpha * math.exp(float(np.clip(self.beta * (v_fraction - 1.0), -30, 30))))

    def center_delta(self, i: int, s_without_i: np.ndarray) -> dict[str, float]:
        k_without = int(s_without_i.sum())
        v_without = float(np.dot(s_without_i, self.vram_fraction))
        queue_cost = self.dcomp(k_without + 1)
        memory_cost = self.barrier(v_without + self.vram_fraction[i]) - self.barrier(v_without)
        return {
            "k_without_i": k_without,
            "candidate_k": k_without + 1,
            "vram_without_i_fraction": v_without,
            "candidate_vram_fraction": v_without + float(self.vram_fraction[i]),
            "center_compute_cost": float(self.center_compute[i]),
            "queue_cost": queue_cost,
            "memory_barrier_increment": memory_cost,
            "center_cost_increment": float(self.center_compute[i] + queue_cost + memory_cost),
        }

    def target(self, i: int, s: np.ndarray) -> tuple[int, dict[str, float]]:
        without = s.copy()
        without[i] = 0
        delta = self.center_delta(i, without)
        local = float(self.a["e_loc"][i])
        offload = float(self.a["e_tx"][i] + delta["center_cost_increment"])
        return int(offload < local), {**delta, "local_cost": local, "offload_cost": offload}

    def potential(self, s: np.ndarray) -> float:
        k = int(s.sum())
        v = float(np.dot(s, self.vram_fraction))
        base = np.where(s == 1, self.a["e_tx"] + self.center_compute, self.a["e_loc"]).sum()
        return float(base + sum(self.dcomp(j) for j in range(1, k + 1)) + self.barrier(v))

    def individual_costs(self, s: np.ndarray) -> np.ndarray:
        result = np.zeros(self.n)
        for i in range(self.n):
            if s[i] == 0:
                result[i] = self.a["e_loc"][i]
            else:
                without = s.copy()
                without[i] = 0
                result[i] = self.a["e_tx"][i] + self.center_delta(i, without)["center_cost_increment"]
        return result

    def total_cost(self, s: np.ndarray) -> float:
        return float(self.individual_costs(s).sum())

    def vram_load_fraction(self, s: np.ndarray) -> float:
        return float(np.dot(s, self.vram_fraction))

    def is_psne(self, s: np.ndarray) -> bool:
        return all(self.target(i, s)[0] == int(s[i]) for i in range(self.n))

    def metrics(self, s: np.ndarray) -> dict[str, Any]:
        k = int(s.sum())
        energy_mj = np.where(s == 1, self.a["tx_energy_mj"], self.a["local_energy_mj"])
        return {
            "k_offload": k,
            "offload_fraction": k / self.n,
            "potential": self.potential(s),
            "system_total_cost_normalized": self.total_cost(s),
            "mean_energy_mj_per_task_simulated": float(energy_mj.mean()),
            "queue_delay_ms_per_offloaded_task": self.dcomp(k) * 1000.0 if k else 0.0,
            "mean_queue_delay_ms_per_task": (k / self.n) * self.dcomp(k) * 1000.0 if k else 0.0,
            "vram_load_gb_simulated": self.vram_load_fraction(s) * self.c.vram_capacity_gb_assumed,
            "vram_load_fraction": self.vram_load_fraction(s),
            "memory_violation_simulated": int(self.vram_load_fraction(s) > 1.0),
            "is_psne": int(self.is_psne(s)),
        }


class DeepSeekGameMaster:
    def __init__(self, client: DeepSeekClient, formula: FormulaModel, cache_path: Path):
        self.client = client
        self.formula = formula
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.events: list[dict[str, Any]] = []
        self.seen_states: set[str] = set()

    def reply(self, iteration: int, i: int, intent: dict[str, Any], s_without_i: np.ndarray) -> dict[str, Any]:
        exact = self.formula.center_delta(i, s_without_i)
        payload = {
            "current_congestion_k": exact["k_without_i"],
            "candidate_congestion_k": exact["candidate_k"],
            "current_vram_load_fraction": round(exact["vram_without_i_fraction"], 6),
            "candidate_vram_load_fraction": round(exact["candidate_vram_fraction"], 6),
            "intent": intent,
            "formula_components": {
                "center_compute_cost": exact["center_compute_cost"],
                "queue_cost": exact["queue_cost"],
                "memory_barrier_increment": exact["memory_barrier_increment"],
            },
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        persistent_cache_hit = key in self.cache
        state_reused = key in self.seen_states
        self.seen_states.add(key)
        if persistent_cache_hit:
            result = dict(self.cache[key])
        else:
            system = (
                "You are the central Game Master in an AI-RAN offloading simulation. Read current congestion "
                "and the structured task Intent. Return JSON only with center_cost_increment, compute_impact, "
                "vram_impact, semantic_warning. center_cost_increment must equal the sum of the three supplied "
                "formula components. Do not choose the node strategy."
            )
            response = self.client.complete_json(system, payload, max_tokens=180)
            raw = response.pop("parsed")
            result = {
                "llm_center_cost_raw": safe_float(raw.get("center_cost_increment"), exact["center_cost_increment"]),
                "compute_impact": str(raw.get("compute_impact", ""))[:200],
                "vram_impact": str(raw.get("vram_impact", ""))[:200],
                "semantic_warning": str(raw.get("semantic_warning", ""))[:300],
                "model": self.client.model,
                **response,
            }
            self.cache[key] = result
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        verified = exact["center_cost_increment"]
        tolerance = 1e-5 * max(1.0, abs(verified))
        event = {
            "iteration": iteration,
            "node": i,
            **exact,
            "llm_center_cost_raw": result["llm_center_cost_raw"],
            "verified_center_cost_increment": verified,
            "llm_numeric_within_tolerance": int(abs(result["llm_center_cost_raw"] - verified) <= tolerance),
            "state_hash": key,
            "state_reused_in_experiment": int(state_reused),
            "persistent_cache_hit": int(persistent_cache_hit),
            **{key: result.get(key) for key in [
                "compute_impact", "vram_impact", "semantic_warning", "model", "latency_ms",
                "resolved_model", "prompt_tokens", "completion_tokens", "total_tokens", "attempts", "parse_ok",
            ]},
        }
        self.events.append(event)
        return event


def run_best_response(
    model: FormulaModel,
    c: Config,
    seed: int,
    intents: list[dict[str, Any]] | None = None,
    coordinator: DeepSeekGameMaster | None = None,
    record_trace: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, model.n, dtype=np.int8)
    idle = 0
    trace: list[dict[str, Any]] = []
    for t in range(c.t_max + 1):
        if record_trace:
            trace.append({
                "iteration": t,
                "k_offload": int(s.sum()),
                "potential": model.potential(s),
                "system_total_cost_normalized": model.total_cost(s),
                "vram_load_fraction": model.vram_load_fraction(s),
                "changed": 0,
            })
        if t == c.t_max:
            break
        if idle >= model.n:
            if model.is_psne(s):
                break
            idle = 0
        i = int(rng.integers(0, model.n))
        if coordinator is None:
            new = model.target(i, s)[0]
        else:
            if intents is None:
                raise ValueError("Intents are required with a live coordinator")
            without = s.copy()
            without[i] = 0
            reply = coordinator.reply(t, i, intents[i], without)
            offload = model.a["e_tx"][i] + reply["verified_center_cost_increment"]
            new = int(offload < model.a["e_loc"][i])
        changed = new != int(s[i])
        s[i] = new
        if record_trace:
            trace[-1]["changed"] = int(changed)
        idle = 0 if changed else idle + 1
        if coordinator is not None and (t + 1) % 25 == 0:
            print(f"DeepSeek game: {t + 1} updates, K={int(s.sum())}, idle={idle}", flush=True)
    return s, trace


def baseline_rows(c: Config, model: FormulaModel, equilibrium: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(c.seed + 1000)
    greedy = ((model.a["e_tx"] + model.center_compute) < model.a["e_loc"]).astype(np.int8)
    cases = [
        ("deepseek_best_response_psne", equilibrium),
        ("all_local", np.zeros(model.n, dtype=np.int8)),
        ("all_offload", np.ones(model.n, dtype=np.int8)),
        ("greedy_local_information_only", greedy),
    ]
    rows = []
    for name, strategy in cases:
        rows.append({"algorithm": name, **model.metrics(strategy)})
    random_metrics = [model.metrics(rng.integers(0, 2, model.n, dtype=np.int8)) for _ in range(c.random_trials)]
    numeric = [key for key, value in random_metrics[0].items() if isinstance(value, (int, float, np.number))]
    row = {"algorithm": "random_p_0.5_mean"}
    row.update({key: float(np.mean([item[key] for item in random_metrics])) for key in numeric})
    row["is_psne"] = "not_applicable"
    rows.append(row)
    return rows


def pareto_rows(model: FormulaModel) -> list[dict[str, Any]]:
    score = model.a["e_loc"] - (model.a["e_tx"] + model.center_compute)
    order = np.argsort(score)[::-1]
    candidates = []
    for k in range(model.n + 1):
        s = np.zeros(model.n, dtype=np.int8)
        s[order[:k]] = 1
        candidates.append({"candidate_k": k, **model.metrics(s)})
    for row in candidates:
        row["pareto_nondominated"] = int(not any(
            other["mean_energy_mj_per_task_simulated"] <= row["mean_energy_mj_per_task_simulated"]
            and other["mean_queue_delay_ms_per_task"] <= row["mean_queue_delay_ms_per_task"]
            and (
                other["mean_energy_mj_per_task_simulated"] < row["mean_energy_mj_per_task_simulated"]
                or other["mean_queue_delay_ms_per_task"] < row["mean_queue_delay_ms_per_task"]
            )
            for other in candidates
        ))
    return candidates


def memory_sweep(
    c: Config,
    arrays: dict[str, np.ndarray],
    semantics: pd.DataFrame,
    queue: dict[str, Any],
    memory: dict[str, Any],
    scheduler: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for n in range(c.sweep_n_min, c.sweep_n_max + 1, c.sweep_n_step):
        model = FormulaModel(c, arrays, semantics, queue, memory, scheduler, n)
        counts = {name: 0 for name in ["deepseek_best_response_psne", "all_local", "all_offload", "random_p_0.5", "greedy_local_information_only"]}
        mean_k = {name: [] for name in counts}
        for trial in range(c.sweep_trials):
            equilibrium, _ = run_best_response(model, c, c.seed + n * 100 + trial, record_trace=False)
            rng = np.random.default_rng(c.seed + n * 1000 + trial)
            strategies = {
                "deepseek_best_response_psne": equilibrium,
                "all_local": np.zeros(n, dtype=np.int8),
                "all_offload": np.ones(n, dtype=np.int8),
                "random_p_0.5": rng.integers(0, 2, n, dtype=np.int8),
                "greedy_local_information_only": ((model.a["e_tx"] + model.center_compute) < model.a["e_loc"]).astype(np.int8),
            }
            for name, strategy in strategies.items():
                counts[name] += int(model.vram_load_fraction(strategy) > 1.0)
                mean_k[name].append(int(strategy.sum()))
        for name in counts:
            rows.append({
                "n_nodes": n,
                "algorithm": name,
                "trials": c.sweep_trials,
                "memory_violation_rate_simulated": counts[name] / c.sweep_trials,
                "mean_k_offload": float(np.mean(mean_k[name])),
                "capacity_gb_assumed": c.vram_capacity_gb_assumed,
                "measurement_status": "software_simulator_proxy_not_physical_oom",
            })
    return rows


def formula_audit(model: FormulaModel, trials: int = 200) -> list[dict[str, Any]]:
    rng = np.random.default_rng(12345)
    errors = []
    changed_moves = 0
    for _ in range(trials):
        s = rng.integers(0, 2, model.n, dtype=np.int8)
        i = int(rng.integers(0, model.n))
        before = s.copy()
        after = s.copy()
        after[i] = 1 - after[i]
        delta_phi = model.potential(after) - model.potential(before)
        before_cost = model.individual_costs(before)[i]
        after_cost = model.individual_costs(after)[i]
        errors.append(abs((after_cost - before_cost) - delta_phi))
        changed_moves += 1
    return [{
        "audit": "exact_potential_identity",
        "trials": changed_moves,
        "max_abs_delta_ci_minus_delta_phi": float(max(errors)),
        "mean_abs_delta_ci_minus_delta_phi": float(np.mean(errors)),
        "passed_at_tolerance_1e-9": int(max(errors) < 1e-9),
        "claim_boundary": "formula consistency only; not global optimality or physical validity",
    }]


def assumptions_rows(c: Config, queue: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in asdict(c).items():
        status = "configuration"
        if key.endswith("assumed") or "assumed" in key:
            status = "explicit_assumption"
        rows.append({"parameter": key, "value": value, "status": status})
    rows.extend([
        {"parameter": "queue_mu", "value": queue["mu_per_second"], "status": "fitted_proxy"},
        {"parameter": "queue_lambda", "value": queue["lambda_per_task_per_second"], "status": "fitted_proxy"},
        {"parameter": "memory_alpha", "value": memory["alpha"], "status": "fitted_from_empirical_tail"},
        {"parameter": "memory_beta", "value": memory["beta"], "status": "fitted_from_empirical_tail"},
    ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run real DeepSeek semantic parsing and live Game Master loop")
    parser.add_argument("--prepare", action="store_true", help="Prepare data fits and intents without API calls")
    args = parser.parse_args()
    if not args.full and not args.prepare:
        parser.error("Choose --prepare or --full")

    c = Config()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "data_manifest.csv", data_manifest())
    pool = load_request_pool(c)
    queue, queue_points = fit_mm1_delay(c)
    memory, memory_points = fit_vram_barrier(c)
    scheduler = scheduler_summary()
    intents, arrays = build_intents(pool, c)
    write_csv(OUT / "semantic_intents.csv", intents)
    write_csv(OUT / "queue_mm1_fit_points.csv", queue_points.to_dict("records"))
    write_csv(OUT / "vram_barrier_fit_points.csv", memory_points.to_dict("records"))
    write_csv(OUT / "scheduler_latency_summary.csv", [scheduler])
    write_csv(OUT / "assumptions_and_parameters.csv", assumptions_rows(c, queue, memory))
    (OUT / "fitted_formula_parameters.json").write_text(
        json.dumps({"queue": queue, "memory": memory, "scheduler": scheduler, "config": asdict(c)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.prepare:
        print(json.dumps({"status": "prepared", "queue": queue, "memory": memory}, ensure_ascii=False, indent=2))
        return

    client = DeepSeekClient(c)
    semantic_parser = SemanticIntentParser(client, OUT / "deepseek_semantic_cache.json")
    predictions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=c.semantic_api_workers) as executor:
        futures = [executor.submit(semantic_parser.evaluate, intent) for intent in intents]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            predictions.append(future.result())
            if index % 20 == 0:
                print(f"DeepSeek semantic parsing: {index}/{len(intents)}", flush=True)
    semantic_df = pd.DataFrame(predictions).sort_values("node").reset_index(drop=True)
    write_csv(OUT / "semantic_resource_predictions.csv", semantic_df.to_dict("records"))

    main_model = FormulaModel(c, arrays, semantic_df, queue, memory, scheduler, c.n_main)
    game_master = DeepSeekGameMaster(client, main_model, OUT / "deepseek_game_master_cache.json")
    equilibrium, trace = run_best_response(
        main_model,
        c,
        c.seed,
        intents=intents[:c.n_main],
        coordinator=game_master,
    )
    write_csv(OUT / "convergence_trace.csv", trace)
    write_csv(OUT / "llm_feedback_events.csv", game_master.events)
    write_csv(OUT / "equilibrium_strategy_s_star.csv", [
        {
            "node": i,
            "s_star": int(equilibrium[i]),
            "meaning": "offload" if equilibrium[i] else "local",
            "vram_gb_simulated": float(main_model.vram_gb[i]),
            "compute_multiplier": float(main_model.compute_multiplier[i]),
            "vram_multiplier": float(main_model.vram_multiplier[i]),
        }
        for i in range(c.n_main)
    ])
    baselines = baseline_rows(c, main_model, equilibrium)
    write_csv(OUT / "algorithm_comparison.csv", baselines)
    write_csv(OUT / "pareto_front.csv", pareto_rows(main_model))
    write_csv(OUT / "memory_violation_sweep.csv", memory_sweep(c, arrays, semantic_df, queue, memory, scheduler))
    write_csv(OUT / "formula_audit.csv", formula_audit(main_model))

    semantic_api_records = list(semantic_parser.cache.values())
    game_api_records = list(game_master.cache.values())
    all_real = semantic_api_records + game_api_records
    overhead = {
        "model": client.model,
        "resolved_models": ";".join(sorted({str(row.get("resolved_model", "")) for row in all_real if row.get("resolved_model")})),
        "semantic_logical_calls": len(predictions),
        "semantic_real_api_calls": len(semantic_api_records),
        "game_logical_calls": len(game_master.events),
        "game_real_api_calls": len(game_api_records),
        "game_cache_hits": len(game_master.events) - len(game_api_records),
        "total_real_api_calls": len(all_real),
        "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in all_real),
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in all_real])) if all_real else 0.0,
        "p95_latency_ms": float(np.quantile([row["latency_ms"] for row in all_real], 0.95)) if all_real else 0.0,
        "game_numeric_mismatches_corrected": sum(not bool(row["llm_numeric_within_tolerance"]) for row in game_master.events),
        "api_key_saved": False,
        "deployment_claim": "measured API latency; no microsecond/millisecond claim unless observed",
    }
    write_csv(OUT / "llm_coordination_overhead.csv", [overhead])

    eq_metrics = main_model.metrics(equilibrium)
    summary = {
        "status": "completed",
        "experiment": "PDF Algorithm 1 plus advisor-required semantic data-driven coordination",
        "n_main": c.n_main,
        "updates": len(trace) - 1,
        "strategy_changes": sum(int(row["changed"]) for row in trace),
        "converged_to_verified_psne": bool(main_model.is_psne(equilibrium)),
        "equilibrium": eq_metrics,
        "queue_fit_r2": queue["r2_seconds"],
        "memory_barrier_fit_r2": memory["r2_tail_pressure"],
        "llm_overhead": overhead,
        "section_3_outputs": {
            "baselines": "algorithm_comparison.csv",
            "convergence_and_stability": "convergence_trace.csv",
            "pareto_and_system_total_cost": "pareto_front.csv and algorithm_comparison.csv",
            "memory_violation_rate_n_30_to_200": "memory_violation_sweep.csv",
            "llm_orchestration_overhead": "llm_coordination_overhead.csv and llm_feedback_events.csv",
        },
        "claim_boundary": "PSNE and proxy-metric results for this finite simulation; not global optimum, physical OOM proof, statistical general law, or deployment validation.",
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
