#!/usr/bin/env python3
"""Generate the final Mark4 figures in PNG, PDF, SVG, and TIFF."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "02_表格数据"
FIGURES = ROOT / "03_结果图"

COLORS = {
    "blue": "#2F6B9A",
    "teal": "#238B8D",
    "red": "#C44E52",
    "gold": "#D39C2C",
    "gray": "#73777B",
    "green": "#4C956C",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.grid": True,
        "grid.color": "#D9DDE1",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for folder, suffix, kwargs in [
        ("PNG", ".png", {"dpi": 300}),
        ("PDF", ".pdf", {}),
        ("SVG", ".svg", {}),
        ("TIFF", ".tiff", {"dpi": 300, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        target = FIGURES / folder / f"{stem}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def convergence_figure() -> None:
    d = pd.read_csv(TABLES / "convergence_trace.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    ax.axvspan(100, 300, color=COLORS["gold"], alpha=0.11, label="PDF expected 1N-3N window")
    ax.plot(d.iteration, d.potential, color=COLORS["blue"], lw=1.8, label="Exact potential $\\Phi(s)$")
    changed = d[d.changed == 1]
    ax.scatter(changed.iteration, changed.potential, s=9, color=COLORS["red"], zorder=3, label="Strategy change")
    ax.set_ylabel("Potential value")
    ax.set_title("A  Convergence and stability of asynchronous best response", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[1]
    ax.step(d.iteration, d.k_offload, where="post", color=COLORS["teal"], lw=1.6, label="Offloaded tasks $K$")
    ax.plot(d.iteration, d.vram_load_fraction * 100, color=COLORS["red"], lw=1.2, alpha=0.85, label="VRAM load (%)")
    ax.axhline(100, color="#222222", lw=0.9, ls="--", label="Capacity threshold")
    ax.set_xlabel("Asynchronous update")
    ax.set_ylabel("Count / percent")
    ax.set_title("B  Strategy and capacity trajectory", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.text(0.99, 0.93, "PSNE verified at update 543\n$K^*=71$, VRAM=91.1%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "square,pad=0.35", "fc": "white", "ec": "#B7BDC3", "lw": 0.7})
    save(fig, "01_convergence_stability")


def pareto_cost_figure() -> None:
    comp = pd.read_csv(TABLES / "algorithm_comparison.csv")
    pareto = pd.read_csv(TABLES / "pareto_front.csv")
    names = {
        "deepseek_best_response_psne": "DeepSeek BR\nPSNE",
        "all_local": "All-Local",
        "all_offload": "All-Offload",
        "greedy_local_information_only": "Greedy",
        "random_p_0.5_mean": "Random",
    }
    order = ["deepseek_best_response_psne", "all_local", "all_offload", "random_p_0.5_mean", "greedy_local_information_only"]
    c = comp.set_index("algorithm").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    colors = [COLORS["teal"], COLORS["gray"], COLORS["red"], COLORS["blue"], COLORS["gold"]]
    bars = ax.bar(range(len(c)), c.system_total_cost_normalized, color=colors, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(c)), [names[x] for x in c.algorithm])
    ax.set_ylabel("Normalized system total cost (log scale)")
    ax.set_title("A  Required baseline comparison", loc="left", fontweight="bold")
    for bar, value in zip(bars, c.system_total_cost_normalized):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,.1f}", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    dominated = pareto[pareto.pareto_nondominated == 0]
    front = pareto[pareto.pareto_nondominated == 1].sort_values("mean_energy_mj_per_task_simulated")
    ax.scatter(dominated.mean_energy_mj_per_task_simulated / 1000,
               dominated.mean_queue_delay_ms_per_task, s=16, color="#C8CDD2", label="Candidate allocation")
    ax.plot(front.mean_energy_mj_per_task_simulated / 1000,
            front.mean_queue_delay_ms_per_task, color=COLORS["blue"], lw=1.5, marker="o", ms=3, label="Nondominated frontier")
    eq = c.iloc[0]
    ax.scatter(eq.mean_energy_mj_per_task_simulated / 1000, eq.mean_queue_delay_ms_per_task,
               s=75, marker="*", color=COLORS["red"], edgecolor="white", linewidth=0.6, zorder=5, label="PSNE")
    ax.set_xlabel("Mean simulated energy (J/task)")
    ax.set_ylabel("Mean queue delay (ms/task)")
    ax.set_title("B  Energy-delay Pareto candidates", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="best")
    save(fig, "02_pareto_system_cost")


def violation_figure() -> None:
    d = pd.read_csv(TABLES / "memory_violation_sweep.csv")
    style = {
        "deepseek_best_response_psne": ("DeepSeek BR PSNE", COLORS["teal"], "o"),
        "all_offload": ("All-Offload", COLORS["red"], "s"),
        "random_p_0.5": ("Random", COLORS["blue"], "^"),
        "greedy_local_information_only": ("Greedy", COLORS["gold"], "D"),
        "all_local": ("All-Local", COLORS["gray"], "v"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, (label, color, marker) in style.items():
        part = d[d.algorithm == key]
        ax.plot(part.n_nodes, part.memory_violation_rate_simulated * 100,
                label=label, color=color, marker=marker, ms=4, lw=1.5)
    ax.set_xlabel("Number of edge nodes $N$")
    ax.set_ylabel("Simulated memory violation rate (%)")
    ax.set_ylim(-3, 105)
    ax.set_xticks(np.arange(30, 201, 10))
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Memory-capacity violations across the PDF-required $N=30$-200 sweep", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.text(0.01, 0.97, "Software simulator, assumed 16 GB capacity\n30 deterministic-seed trials per N; not physical RTX 4080 OOM",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "square,pad=0.35", "fc": "white", "ec": "#B7BDC3", "lw": 0.7})
    save(fig, "03_memory_violation_rate")


def overhead_figure() -> None:
    events = pd.read_csv(TABLES / "llm_feedback_events.csv")
    semantic = pd.read_csv(TABLES / "semantic_resource_predictions.csv")
    overhead = pd.read_csv(TABLES / "llm_coordination_overhead.csv").iloc[0]
    game_unique = events.drop_duplicates("state_hash")
    latencies = pd.concat([semantic.latency_ms, game_unique.latency_ms], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.hist(latencies / 1000, bins=28, color=COLORS["blue"], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(overhead.mean_latency_ms / 1000, color=COLORS["red"], lw=1.5, label=f"Mean {overhead.mean_latency_ms/1000:.2f} s")
    ax.axvline(overhead.p95_latency_ms / 1000, color=COLORS["gold"], lw=1.5, ls="--", label=f"P95 {overhead.p95_latency_ms/1000:.2f} s")
    ax.set_xlabel("Real API latency (seconds)")
    ax.set_ylabel("API records")
    ax.set_title("A  DeepSeek coordination latency", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    labels = ["Semantic\nlogical", "Game\nlogical", "Real API\nrecords", "Game state\nreuse"]
    values = [overhead.semantic_logical_calls, overhead.game_logical_calls,
              overhead.total_real_api_calls, overhead.game_cache_hits]
    bars = ax.bar(labels, values, color=[COLORS["teal"], COLORS["blue"], COLORS["red"], COLORS["gray"]], width=0.68)
    ax.set_ylabel("Number of calls / states")
    ax.set_title("B  Coordination volume", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 10, f"{int(value):,}", ha="center", fontsize=8)
    ax.text(0.02, 0.96, f"Tokens: {int(overhead.total_tokens):,}\nResolved model: {overhead.resolved_models}",
            transform=ax.transAxes, va="top", fontsize=8)
    save(fig, "04_llm_orchestration_overhead")


def calibration_figure() -> None:
    q = pd.read_csv(TABLES / "queue_mm1_fit_points.csv")
    v = pd.read_csv(TABLES / "vram_barrier_fit_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.scatter(q.k_equivalent, q.queue_delay_ms, s=9, alpha=0.22, color=COLORS["gray"], label="Matched gateway observations")
    fit = q.groupby("k_equivalent").queue_fit_ms.mean().reset_index().sort_values("k_equivalent")
    ax.plot(fit.k_equivalent, fit.queue_fit_ms, color=COLORS["red"], lw=1.8, label="M/M/1 robust fit")
    ax.set_xlabel("Equivalent congestion $K$ (scaled proxy)")
    ax.set_ylabel("Queue delay (ms)")
    ax.set_title("A  Queue formula calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.text(0.97, 0.95, "$R^2=-0.012$\nWeak queue-delay association",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ax = axes[1]
    ax.scatter(v.vram_utilization_proxy * 100, v.empirical_tail_pressure, s=18, color=COLORS["blue"], alpha=0.7, label="Empirical tail pressure")
    ax.plot(v.vram_utilization_proxy * 100, v.barrier_fit, color=COLORS["red"], lw=1.8, label="Exponential barrier fit")
    ax.set_yscale("log")
    ax.axvline(100, color="#222222", ls="--", lw=0.9)
    ax.set_xlabel("VRAM utilization relative to observed P99 (%)")
    ax.set_ylabel("Normalized memory pressure (log scale)")
    ax.set_title("B  VRAM barrier calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.text(0.03, 0.95, "$R^2=0.992$\n$\\beta=11.01$",
            transform=ax.transAxes, va="top", fontsize=8)
    save(fig, "05_formula_calibration")


def main() -> None:
    configure()
    convergence_figure()
    pareto_cost_figure()
    violation_figure()
    overhead_figure()
    calibration_figure()
    print(f"Wrote final figures to {FIGURES}")


if __name__ == "__main__":
    main()
