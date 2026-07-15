#!/usr/bin/env python3
"""Publication-grade figure exports for Mark1 and Mark2 (Python-only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#0F4D92"
BLUE_2 = "#3775BA"
TEAL = "#42949E"
RED = "#B64342"
GREY = "#767676"
LIGHT = "#CFCECE"
VIOLET = "#9A4D8E"
GREEN = "#2E7D4F"
BLACK = "#272727"

METHOD_LABELS = {
    "best_response_psne": "Best response\n(PSNE)",
    "all_local": "All local",
    "all_offload": "All offload",
    "random_p_0.5_mean": "Random\n(mean)",
    "greedy_energy_only": "Greedy",
    "greedy_trace_base_only": "Greedy",
    "social_optimum_diagnostic": "Social optimum\n(diagnostic)",
}

METHOD_COLORS = {
    "best_response_psne": BLUE,
    "all_local": GREY,
    "all_offload": RED,
    "random_p_0.5_mean": LIGHT,
    "greedy_energy_only": VIOLET,
    "greedy_trace_base_only": VIOLET,
    "social_optimum_diagnostic": GREEN,
}


def save_bundle(fig: plt.Figure, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    formats = {
        ".svg": ("SVG", {}), ".pdf": ("PDF", {}),
        ".png": ("PNG", {"dpi": 300}), ".tiff": ("TIFF", {"dpi": 600}),
    }
    for suffix, (folder, kwargs) in formats.items():
        target = stem.parent / folder / f"{stem.name}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def convergence(data: Path, out: Path, mark: str):
    df = pd.read_csv(data / "convergence_trace.csv")
    changed = df[df.changed == 1]
    fig, ax = plt.subplots(figsize=(3.50, 2.35))
    ax.plot(df.iteration, df.potential, color=BLUE, lw=1.6, zorder=2)
    ax.scatter(changed.iteration, changed.potential, s=5, color=TEAL, alpha=.55,
               edgecolors="none", zorder=3, label="Strategy change")
    ax.scatter([df.iteration.iloc[-1]], [df.potential.iloc[-1]], s=25, facecolor="white",
               edgecolor=RED, linewidth=.9, zorder=4)
    ax.annotate(f"PSNE\nK = {int(df.k_offload.iloc[-1])}",
                xy=(df.iteration.iloc[-1], df.potential.iloc[-1]), xytext=(-30, 17),
                textcoords="offset points", ha="right", va="bottom", color=RED,
                arrowprops={"arrowstyle": "-", "color": RED, "lw": .7})
    ax.set(xlabel="Asynchronous update", ylabel=r"Potential, $\Phi(\mathbf{s})$")
    ax.text(.02, .04, f"{int(changed.shape[0])} strategy changes",
            transform=ax.transAxes, color=GREY, fontsize=6.5)
    ax.legend(loc="upper right", handletextpad=.4)
    ax.set_title(f"{mark}: monotonic convergence to a stable equilibrium", loc="left", pad=5)
    ax.margins(x=.02)
    save_bundle(fig, out / "convergence")


def strategy(data: Path, out: Path, mark: str):
    df = pd.read_csv(data / "equilibrium_strategy_s_star.csv")
    s = df.s_star.to_numpy(int)
    fig, (ax, ax_sum) = plt.subplots(1, 2, figsize=(3.50, 1.75),
                                     gridspec_kw={"width_ratios": [4.5, 1], "wspace": .18})
    matrix = s.reshape(10, 10)
    cmap = mpl.colors.ListedColormap([LIGHT, BLUE])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_xticks([0, 4, 9], ["1", "5", "10"])
    ax.set_yticks([0, 4, 9], ["1", "5", "10"])
    ax.set_xlabel("Request column")
    ax.set_ylabel("Request row")
    for spine in ax.spines.values(): spine.set_visible(False)
    k = int(s.sum()); local = len(s) - k
    bars = ax_sum.bar([0, 1], [k, local], color=[BLUE, LIGHT], width=.68,
                      edgecolor=BLACK, linewidth=.45)
    ax_sum.set_xticks([0, 1], ["Offload", "Local"], rotation=35, ha="right")
    ax_sum.set_ylabel("Requests")
    ax_sum.set_ylim(0, max(k, local) * 1.18)
    for bar, val in zip(bars, [k, local]):
        ax_sum.text(bar.get_x() + bar.get_width()/2, val + 2, str(val), ha="center", va="bottom")
    fig.suptitle(mark + r": equilibrium strategy $\mathbf{s}^*$", x=.08, ha="left", fontsize=8, y=1.02)
    save_bundle(fig, out / "equilibrium_strategy_s_star")


def comparison(data: Path, out: Path, mark: str):
    df = pd.read_csv(data / "algorithm_comparison.csv")
    names = df.algorithm.tolist(); values = df.system_total_cost.to_numpy(float)
    labels = [METHOD_LABELS.get(n, n) for n in names]
    colors = [METHOD_COLORS.get(n, GREY) for n in names]
    fig, ax = plt.subplots(figsize=(3.50, 2.45))
    bars = ax.bar(np.arange(len(names)), values, color=colors, width=.72,
                  edgecolor=BLACK, linewidth=.45)
    ax.set_yscale("log")
    ax.set_ylim(values.min() * 0.78, values.max() * 1.75)
    ax.set_ylabel("System total cost (log scale)")
    ax.set_xticks(np.arange(len(names)), labels, rotation=30, ha="right")
    ax.set_title(f"{mark}: equilibrium stability does not imply social optimality", loc="left", pad=5)
    for bar, value in zip(bars, values):
        label = f"{value:.1f}" if value < 1000 else f"{value/1000:.1f}k"
        ax.text(bar.get_x()+bar.get_width()/2, value*1.12, label, ha="center", va="bottom", fontsize=6)
    eq = float(df.loc[df.algorithm == "best_response_psne", "system_total_cost"].iloc[0])
    opt = float(df.loc[df.algorithm == "social_optimum_diagnostic", "system_total_cost"].iloc[0])
    ax.text(.02, .78, f"Equilibrium / optimum = {eq/opt:.2f}", transform=ax.transAxes,
            ha="left", va="top", color=BLACK, fontsize=6.5)
    ax.margins(x=.04)
    save_bundle(fig, out / "algorithm_comparison")


def violation(data: Path, out: Path, mark: str):
    df = pd.read_csv(data / "memory_violation_sweep.csv")
    xcol = "n" if "n" in df.columns else "capacity_k"
    label_x = "Number of edge nodes, N" if xcol == "n" else "Assumed capacity, K"
    styles = {
        "best_response_psne": (BLUE, "o", "Best response"),
        "greedy_energy_only": (VIOLET, "s", "Greedy"),
        "greedy_trace_base_only": (VIOLET, "s", "Greedy"),
        "random_p_0.5": (GREY, "^", "Random"),
    }
    fig, ax = plt.subplots(figsize=(3.50, 2.30))
    ordered = [x for x in styles if x in set(df.algorithm)]
    endpoints = []
    for name in ordered:
        part = df[df.algorithm == name].sort_values(xcol)
        color, marker, label = styles[name]
        ax.plot(part[xcol], part.violation_rate, color=color, lw=1.35, marker=marker,
                ms=3.2, mfc="white", mec=color, mew=.7)
        endpoints.append((float(part[xcol].iloc[-1]), float(part.violation_rate.iloc[-1]), label, color))
    ax.set(xlabel=label_x, ylabel="Memory violation rate")
    ax.set_ylim(-.04, 1.08); ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_title(f"{mark}: congestion-aware responses prevent capacity violations", loc="left", pad=5)
    # Direct labels are placed at stable series endpoints; offsets avoid overlap.
    offsets = {"Best response": -9, "Greedy": 7, "Random": -1}
    for x, y, label, color in endpoints:
        ax.annotate(label, (x, y), xytext=(5, offsets[label]), textcoords="offset points",
                    color=color, va="center", fontsize=6.5, clip_on=False)
    ax.margins(x=.04)
    save_bundle(fig, out / "memory_violation_sweep")


def render_mark(folder: Path, mark: str):
    data = folder / "02_表格数据"; out = folder / "03_结果图"
    convergence(data, out, mark)
    strategy(data, out, mark)
    comparison(data, out, mark)
    violation(data, out, mark)


if __name__ == "__main__":
    mark_dir = Path(__file__).resolve().parents[1]
    render_mark(mark_dir, "Mark2 Alibaba")
