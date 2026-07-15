#!/usr/bin/env python3
"""Mark2: Alibaba production-trace-driven AI-RAN offloading simulation.

Trace-derived inputs and modeling assumptions are explicitly separated.
Best response is evaluated as convergence to a PSNE, not global optimality.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARK_DIR = Path(__file__).resolve().parents[1]
TRACE = MARK_DIR / "00_原始数据" / "阿里_GenAI服务_2026"
OUT = MARK_DIR / "02_表格数据"
FIG = MARK_DIR / "03_结果图"


@dataclass(frozen=True)
class Config:
    seed: int = 20260713
    n: int = 100
    peak_hour: str = "2024-12-06 00:00:00"
    t_max: int = 5000
    no_change_stop: int = 100
    local_slowdown: float = 1.8
    local_energy_addition: float = 0.4
    tx_base_cost: float = 0.15
    prompt_tx_weight: float = 0.15
    capacity_fraction: float = 0.80
    memory_alpha: float = 1.2
    memory_beta: float = 8.0
    random_trials: int = 500
    sweep_trials: int = 100


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def load_trace(c: Config):
    req = pd.read_csv(TRACE / "lora_request_trace.csv")
    req["gmt_create"] = pd.to_datetime(req["gmt_create"])
    start = pd.Timestamp(c.peak_hour)
    sample = req[(req.predict_status == "SUCCEED") &
                 (req.gmt_create >= start) & (req.gmt_create < start + pd.Timedelta(hours=1))].copy()
    sample = sample[sample.exec_time_seconds > 0]
    if len(sample) < c.n:
        raise ValueError(f"Peak window has only {len(sample)} usable requests")
    sample = sample.sample(c.n, random_state=c.seed).sort_values("gmt_create").reset_index(drop=True)

    queue = pd.read_csv(TRACE / "queue_rt_raw_anon.csv")["value"]
    infer = pd.read_csv(TRACE / "model_predict_data_anon.csv")["value"]
    gpu_mem = pd.read_csv(TRACE / "pod_gpu_memory_used_bytes_anon.csv")["value"]
    gpu_util = pd.read_csv(TRACE / "pod_gpu_duty_cycle_anon.csv")["value"]
    calibration = {
        "queue_median_ms": float(queue.median()),
        "queue_p95_ms": float(queue.quantile(.95)),
        "inference_median_ms": float(infer.median()),
        "gpu_memory_median_bytes": float(gpu_mem.median()),
        "gpu_memory_p95_bytes": float(gpu_mem.quantile(.95)),
        "gpu_util_median_percent": float(gpu_util.median()),
        "gpu_util_p95_percent": float(gpu_util.quantile(.95)),
        "peak_window_available_success_requests": int(len(req[(req.predict_status == "SUCCEED") &
            (req.gmt_create >= start) & (req.gmt_create < start + pd.Timedelta(hours=1))])),
    }
    return sample, calibration


class Model:
    def __init__(self, sample: pd.DataFrame, calibration: dict, c: Config):
        self.c = c
        med_exec = float(sample.exec_time_seconds.median())
        self.exec_norm = sample.exec_time_seconds.to_numpy(float) / med_exec
        prompt = sample.prompt_length.fillna(sample.prompt_length.median()).to_numpy(float)
        prompt_scale = max(float(np.quantile(prompt, .95)), 1.0)
        self.e_tx = c.tx_base_cost + c.prompt_tx_weight * np.minimum(prompt / prompt_scale, 1.5)
        self.e_loc = c.local_slowdown * self.exec_norm + c.local_energy_addition
        self.queue_base = calibration["queue_median_ms"] / calibration["inference_median_ms"]
        self.queue_ratio = calibration["queue_p95_ms"] / calibration["queue_median_ms"]
        self.k_capacity = int(math.floor(c.capacity_fraction * c.n))

    def congestion(self, k: int) -> float:
        load = k / self.c.n
        queue_cost = self.queue_base * (1.0 + (self.queue_ratio - 1.0) * load * load)
        memory_cost = self.c.memory_alpha * math.exp(self.c.memory_beta * (k / self.k_capacity - 1.0))
        return queue_cost + memory_cost

    def potential(self, s: np.ndarray) -> float:
        k = int(s.sum())
        energy = np.where(s == 1, self.e_tx + self.exec_norm, self.e_loc).sum()
        return float(energy + sum(self.congestion(j) for j in range(1, k + 1)))

    def total_cost(self, s: np.ndarray) -> float:
        k = int(s.sum())
        base = np.where(s == 1, self.e_tx + self.exec_norm, self.e_loc).sum()
        return float(base + k * self.congestion(k))

    def is_psne(self, s: np.ndarray, tol=1e-12) -> bool:
        k = int(s.sum())
        for i in range(self.c.n):
            km = k - int(s[i])
            local = self.e_loc[i]
            offload = self.e_tx[i] + self.exec_norm[i] + self.congestion(km + 1)
            current, alt = (offload, local) if s[i] else (local, offload)
            if alt < current - tol:
                return False
        return True

    def best_response(self, rng: np.random.Generator):
        s = rng.integers(0, 2, self.c.n, dtype=np.int8)
        idle = 0; trace = []
        for t in range(self.c.t_max + 1):
            trace.append({"iteration": t, "k_offload": int(s.sum()),
                          "potential": self.potential(s), "system_total_cost": self.total_cost(s),
                          "changed": 0})
            if t == self.c.t_max:
                break
            if idle >= self.c.no_change_stop:
                if self.is_psne(s):
                    break
                idle = 0
            i = int(rng.integers(0, self.c.n))
            km = int(s.sum()) - int(s[i])
            new = int(self.e_tx[i] + self.exec_norm[i] + self.congestion(km + 1) < self.e_loc[i])
            changed = new != int(s[i]); s[i] = new
            trace[-1]["changed"] = int(changed)
            idle = 0 if changed else idle + 1
        return s, trace

    def social_optimum(self):
        gaps = self.e_tx + self.exec_norm - self.e_loc
        order = np.argsort(gaps)
        best = None
        for k in range(self.c.n + 1):
            s = np.zeros(self.c.n, dtype=np.int8); s[order[:k]] = 1
            value = self.total_cost(s)
            if best is None or value < best[0]: best = (value, s.copy())
        return best[1]

    def metrics(self, name: str, s: np.ndarray):
        k = int(s.sum())
        return {"algorithm": name, "n": self.c.n, "k_offload": k,
                "offload_fraction": k / self.c.n,
                "trace_based_mean_base_cost": float(np.where(s == 1, self.e_tx + self.exec_norm, self.e_loc).mean()),
                "shared_congestion_cost": self.congestion(k),
                "memory_capacity_k_assumed": self.k_capacity,
                "memory_violation": int(k > self.k_capacity),
                "potential": self.potential(s), "system_total_cost": self.total_cost(s),
                "is_psne": int(self.is_psne(s))}


def main():
    c = Config(); OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    sample, calibration = load_trace(c); model = Model(sample, calibration, c)
    rng = np.random.default_rng(c.seed)
    eq, trace = model.best_response(rng); optimum = model.social_optimum()
    all_local = np.zeros(c.n, dtype=np.int8); all_off = np.ones(c.n, dtype=np.int8)
    greedy = ((model.e_tx + model.exec_norm) < model.e_loc).astype(np.int8)
    rows = [model.metrics("best_response_psne", eq), model.metrics("all_local", all_local),
            model.metrics("all_offload", all_off)]
    random_rows = [model.metrics("random_p_0.5", rng.integers(0, 2, c.n, dtype=np.int8))
                   for _ in range(c.random_trials)]
    random_mean = {k: ("random_p_0.5_mean" if k == "algorithm" else
                       float(np.mean([r[k] for r in random_rows]))) for k in rows[0]}
    rows += [random_mean, model.metrics("greedy_trace_base_only", greedy),
             model.metrics("social_optimum_diagnostic", optimum)]

    input_rows=[]
    for i, r in sample.iterrows():
        input_rows.append({"node": i, "gmt_create": r.gmt_create, "predict_type": r.predict_type,
                           "exec_time_seconds_trace": r.exec_time_seconds,
                           "prompt_length_trace": r.prompt_length, "num_inference_steps_trace": r.num_inference_steps,
                           "num_lora_trace": r.num_lora, "e_loc_assumed_mapped": model.e_loc[i],
                           "e_tx_assumed_mapped": model.e_tx[i], "equilibrium_strategy": int(eq[i]),
                           "social_optimum_strategy": int(optimum[i])})
    strategy = [{"node": i, "s_star": int(eq[i]), "meaning": "offload" if eq[i] else "local"}
                for i in range(c.n)]
    sweep=[]
    for cap_fraction in np.arange(.50, .96, .05):
        cfg = Config(capacity_fraction=round(float(cap_fraction), 2))
        m = Model(sample, calibration, cfg); violations={"best_response_psne":0,"greedy_trace_base_only":0,"random_p_0.5":0}
        for _ in range(c.sweep_trials):
            s,_=m.best_response(rng)
            cases={"best_response_psne":s,"greedy_trace_base_only":((m.e_tx+m.exec_norm)<m.e_loc).astype(np.int8),
                   "random_p_0.5":rng.integers(0,2,c.n,dtype=np.int8)}
            for name,x in cases.items(): violations[name]+=int(int(x.sum())>m.k_capacity)
        for name,count in violations.items(): sweep.append({"capacity_fraction":cfg.capacity_fraction,
            "capacity_k":m.k_capacity,"algorithm":name,"violation_rate":count/c.sweep_trials})

    write_csv(OUT/"equilibrium_strategy_s_star.csv",strategy)
    write_csv(OUT/"convergence_trace.csv",trace)
    write_csv(OUT/"algorithm_comparison.csv",rows)
    write_csv(OUT/"memory_violation_sweep.csv",sweep)
    write_csv(OUT/"trace_input_sample.csv",input_rows)
    write_csv(OUT/"llm_coordination_overhead.csv",[{"coordinator":"numeric_trace_driven_proxy",
        "coordination_calls":len(trace)-1,"strategy_changes":sum(int(r["changed"]) for r in trace),
        "llm_model":"not_used","llm_token_count":"not_measured","llm_inference_latency_ms":"not_measured"}])
    with (OUT/"parameters_and_calibration.json").open("w",encoding="utf-8") as f:
        json.dump({"assumptions":asdict(c),"trace_calibration":calibration},f,indent=2,ensure_ascii=False)
    summary={"converged_to_psne":bool(model.is_psne(eq)),"updates_recorded":len(trace)-1,
             "strategy_changes":sum(int(r["changed"]) for r in trace),"equilibrium_k":int(eq.sum()),
             "equilibrium_cost":model.total_cost(eq),"assumed_capacity_k":model.k_capacity,
             "memory_violation":bool(int(eq.sum())>model.k_capacity),"diagnostic_global_optimum_k":int(optimum.sum()),
             "diagnostic_global_optimum_cost":model.total_cost(optimum),
             "equilibrium_is_global_optimum":bool(np.array_equal(eq,optimum)),
             "claim":"Alibaba-trace-driven best response converged to a PSNE; no global-optimality or deployment claim."}
    with (OUT/"run_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)

    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == "__main__": main()
