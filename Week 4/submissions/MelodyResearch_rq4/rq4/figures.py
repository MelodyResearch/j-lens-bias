"""Figures rebuilt solely from saved results (raw JSONL -> analysis tables -> PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as A  # noqa: E402
from . import config as C  # noqa: E402

COL = {"mc": "#0072B2", "oe": "#D55E00", "diff": "#222222"}   # Okabe-Ito, colour-blind safe


def _ci_point(ax, x, vals, color, label=None, jitter=0.0):
    s = A.bootstrap_mean_ci(np.asarray(vals, float))
    if s["n"] == 0:
        return
    rng = np.random.default_rng(0)
    ax.scatter(x + rng.uniform(-jitter, jitter, len(vals)), vals, s=14, color=color, alpha=0.35, zorder=2)
    ax.errorbar([x], [s["mean"]], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="o", color=color,
                capsize=3, ms=6, zorder=3, label=label)


def make_figures(results_jsonl: Path = C.RAW_JSONL, out_dir: Path = C.FIG_DIR) -> list[Path]:
    df = A.load_results(results_jsonl)
    if df.empty:
        raise ValueError("no results to plot")
    pairs = A.per_pair_effects(df)
    items = A.per_item(pairs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Fig 1: continuous primary. Excess |margin change| (target - mean of 3 random seeds), by format and MC-open.
    fig, axes = plt.subplots(1, len(C.LAYERS), figsize=(11, 3.8), sharey=True)
    for ax, layer in zip(axes, C.LAYERS):
        g = items[items.layer == layer]
        for x, col, key in ((0, "mc_excess_abs", "mc"), (1, "oe_excess_abs", "oe"), (2, "diff_excess_abs", "diff")):
            if col in g:
                _ci_point(ax, x, g[col].dropna().values, COL[key], jitter=0.08)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks([0, 1, 2], ["MC", "open", "MC − open"])
        ax.set_title(f"L{layer}  (n items = {g['diff_excess_abs'].notna().sum() if 'diff_excess_abs' in g else 0})")
    axes[0].set_ylabel("target − mean(random)  in  |Δ log-prob margin|")
    fig.suptitle("Excess absolute teacher-forced margin change (points: base items; bar: mean, 95% clustered bootstrap)", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig1_excess_margin_change.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Fig 2: co-primary behavioural: excess change in classified generated answers.
    fig, axes = plt.subplots(1, len(C.LAYERS), figsize=(11, 3.8), sharey=True)
    for ax, layer in zip(axes, C.LAYERS):
        g = items[items.layer == layer]
        for x, col, key in ((0, "mc_excess_changed", "mc"), (1, "oe_excess_changed", "oe"), (2, "diff_excess_changed", "diff")):
            if col in g:
                _ci_point(ax, x, g[col].dropna().values, COL[key], jitter=0.08)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks([0, 1, 2], ["MC", "open", "MC − open"])
        ax.set_title(f"L{layer}")
    axes[0].set_ylabel("P(category changed | target) − P(changed | random)")
    fig.suptitle("Excess change in classified generated answers (includes hedged / unparseable)", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig2_excess_behavioural_change.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Fig 3: category composition per arm.
    rates = A.category_rates(df)
    cats = ["stereotype", "counter", "unknown_hedged", "unparseable"]
    shades = {"stereotype": "#D55E00", "counter": "#0072B2", "unknown_hedged": "#999999", "unparseable": "#F0E442"}
    fig, axes = plt.subplots(len(C.FORMATS), len(C.LAYERS), figsize=(12, 6), sharey=True)
    for i, fmt in enumerate(C.FORMATS):
        for j, layer in enumerate(C.LAYERS):
            ax = axes[i][j]
            arms = [C.CLEAN] + list(C.ARMS_INTERVENTION)
            bottom = np.zeros(len(arms))
            for cat in cats:
                vals = []
                for arm in arms:
                    lay = -1 if arm == C.CLEAN else layer
                    r = rates[(rates.prompt_format == fmt) & (rates.layer == lay) & (rates.arm == arm) & (rates.category == cat)]
                    vals.append(float(r.rate.iloc[0]) if len(r) else 0.0)
                ax.bar(range(len(arms)), vals, bottom=bottom, color=shades[cat], label=cat if (i, j) == (0, 0) else None)
                bottom += np.array(vals)
            ax.set_xticks(range(len(arms)), [a.replace("random_s", "r") for a in arms], rotation=60, fontsize=7)
            ax.set_title(f"{fmt} L{layer}", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "fig3_category_composition.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths
