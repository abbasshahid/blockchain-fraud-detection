"""Paper figures, drawn one panel per file.

Each panel is written as its own PDF and the panels are grouped side by side in
the manuscript with LaTeX minipages.  Drawing them separately keeps every panel
on an identical canvas and lets the panel labels be typeset in the document
font rather than by matplotlib.

Three rules keep the set consistent and legible:

1. **One canvas for every panel.**  ``PANEL_W`` is half the LNCS text width, so
   a panel included at ``width=\\linewidth`` inside a ``0.49\\linewidth``
   minipage is reproduced at 1:1.  Nothing is rescaled, so the point sizes set
   in :mod:`blockchain_fraud.analysis.plot_style` are the point sizes that reach
   the page.
2. **Fixed axes margins.**  Margins are set explicitly rather than by
   ``tight_layout``, so the plot boxes of two panels placed side by side line up
   exactly instead of drifting with the length of the tick labels.
3. **Legends never cover data.**  Each panel reserves an empty band for its
   legend by extending an axis limit, so the legend sits beside the curves
   rather than on top of them, and every panel with more than one series carries
   its own legend.

Panels produced here:

``fig09a_threshold_sensitivity`` / ``fig09b_reliability``
    The decision-threshold and calibration views of the same defect.
``fig10a_evidence_coverage`` / ``fig10b_deletion_curve``
    Citation coverage (a property of the report) and feature-deletion
    faithfulness (a property of the detector).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blockchain_fraud.analysis.plot_style import (
    LNCS_TEXT_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    apply_inline_style,
    save_figure,
)
from blockchain_fraud.utils.io import read_json

MODELS = ["xgboost", "graphsage", "gcn", "gat"]

# Half the text width, matching a 0.49\linewidth minipage.
PANEL_W = LNCS_TEXT_WIDTH_IN * 0.49
PANEL_H = 1.48

# Identical axes box in every panel. The margins are reserved in inches rather
# than as fractions of the canvas: a fractional margin shrinks with the panel
# and crops the axis labels once the panel gets short.
LEFT_IN, RIGHT_PAD_IN = 0.50, 0.13
BOTTOM_IN, TOP_PAD_IN = 0.38, 0.05
MARGINS = {
    "left": LEFT_IN / PANEL_W,
    "right": 1.0 - RIGHT_PAD_IN / PANEL_W,
    "bottom": BOTTOM_IN / PANEL_H,
    "top": 1.0 - TOP_PAD_IN / PANEL_H,
}

METHOD_STYLE = {
    "abs_standardized": ("proxy (v1)", MODEL_COLORS["gat"], "o"),
    "grad_x_input": ("gradient $\\times$ input (v2)", MODEL_COLORS["graphsage"], "s"),
}

LEGEND_KW = dict(
    fontsize=5.8,
    handlelength=1.5,
    handletextpad=0.45,
    borderpad=0.28,
    labelspacing=0.22,
    borderaxespad=0.25,
    framealpha=0.94,
)


def _panel() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(**MARGINS)
    return fig, ax


# --------------------------------------------------------------------------
# Operating point and calibration
# --------------------------------------------------------------------------
def build_threshold_panel(output_dir: Path) -> Path | None:
    sweep_path = output_dir / "tables" / "threshold_sweep_mean.csv"
    if not sweep_path.exists():
        return None
    sweep = pd.read_csv(sweep_path)
    tuned_path = output_dir / "tables" / "table06_threshold_sensitivity.csv"
    tuned = pd.read_csv(tuned_path) if tuned_path.exists() else pd.DataFrame()

    apply_inline_style()
    fig, ax = _panel()
    for model in MODELS:
        part = sweep[sweep["model"] == model].sort_values("threshold")
        if part.empty:
            continue
        ax.plot(part["threshold"], part["f1"], color=MODEL_COLORS[model], label=MODEL_LABELS[model], zorder=3)
        row = tuned[tuned["model"] == model] if not tuned.empty else pd.DataFrame()
        if not row.empty:
            ax.plot(
                [float(row["tau_star_mean"].iloc[0])],
                [float(row["test_f1_at_tau_star_mean"].iloc[0])],
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markersize=4.6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=4,
            )
    ax.axvline(0.5, color="0.45", linestyle=(0, (1, 2.5)), linewidth=1.0, zorder=1)
    ax.annotate(
        r"$\tau=0.5$",
        xy=(0.5, 0.0),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=5.8,
        color="0.3",
        zorder=5,
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Illicit F1")
    ax.set_xlim(0, 1)
    # Headroom above the highest curve (~0.72) reserved for the legend.
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, **LEGEND_KW)
    save_figure(fig, output_dir / "figures", "fig09a_threshold_sensitivity", tight=False)
    return output_dir / "figures" / "fig09a_threshold_sensitivity.pdf"


def build_reliability_panel(output_dir: Path) -> Path | None:
    bins_path = output_dir / "tables" / "calibration_reliability_bins.csv"
    if not bins_path.exists():
        return None
    bins = pd.read_csv(bins_path)

    apply_inline_style()
    fig, ax = _panel()
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0, zorder=1)
    ax.annotate(
        "perfect", xy=(0.80, 0.80), xytext=(-2, 4), textcoords="offset points",
        fontsize=5.6, color="0.4", rotation=34, zorder=2,
    )
    for model in MODELS:
        raw = bins[(bins["model"] == model) & (bins["variant"] == "raw")].dropna(subset=["mean_predicted"])
        cal = bins[(bins["model"] == model) & (bins["variant"] == "platt_recalibrated")].dropna(subset=["mean_predicted"])
        ax.plot(
            raw["mean_predicted"], raw["observed_frequency"],
            color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], markersize=2.6,
            linewidth=1.3, label=MODEL_LABELS[model], zorder=3,
        )
        ax.plot(
            cal["mean_predicted"], cal["observed_frequency"],
            color=MODEL_COLORS[model], linestyle=":", linewidth=1.15, alpha=0.9, zorder=2,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.30)  # band above the diagonal reserved for the legend
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, **LEGEND_KW)
    save_figure(fig, output_dir / "figures", "fig09b_reliability", tight=False)
    return output_dir / "figures" / "fig09b_reliability.pdf"


# --------------------------------------------------------------------------
# Explanation quality
# --------------------------------------------------------------------------
# The configurations reported in the explanation table, in the same order.
COVERAGE_ROWS = [
    ("grounded_llm", "openrouter", "v1", "Grounded\nBTREP v1", "grounded"),
    ("unconstrained_llm", "openrouter", "v1", "Unconstr.\nBTREP v1", "unconstrained"),
    ("grounded_llm", "gemini", "btrep_v2", "Grounded\nBTREP v2", "grounded"),
]


def build_coverage_panel(output_dir: Path) -> Path | None:
    summary_path = output_dir / "metrics" / "explanation_summary.json"
    if not summary_path.exists():
        return None
    methods = read_json(summary_path).get("methods", [])
    index = {(m["method"], m["provider"], m["evidence_version"]): m for m in methods}

    labels: list[str] = []
    values: list[float] = []
    kinds: list[str] = []
    for method, provider, version, label, kind in COVERAGE_ROWS:
        row = index.get((method, provider, version))
        if row is None or row.get("evidence_coverage") is None:
            continue
        labels.append(label)
        values.append(float(row["evidence_coverage"]))
        kinds.append(kind)
    if not labels:
        return None

    apply_inline_style()
    fig, ax = _panel()
    palette = {"grounded": MODEL_COLORS["graphsage"], "unconstrained": MODEL_COLORS["gat"]}
    positions = np.arange(len(labels))
    for pos, value, kind in zip(positions, values, kinds):
        ax.bar(pos, value, width=0.62, color=palette[kind], zorder=3)
        ax.annotate(
            f"{value:.3f}", xy=(pos, value), xytext=(0, 2.5), textcoords="offset points",
            ha="center", fontsize=5.8, color="0.15", zorder=4,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette["grounded"]),
        plt.Rectangle((0, 0), 1, 1, color=palette["unconstrained"]),
    ]
    ax.legend(handles, ["evidence-grounded", "unconstrained"], loc="upper right", **LEGEND_KW)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5.9)
    ax.set_ylabel("Evidence coverage")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.62, len(labels) - 0.38)
    ax.grid(axis="x", visible=False)
    save_figure(fig, output_dir / "figures", "fig10a_evidence_coverage", tight=False)
    return output_dir / "figures" / "fig10a_evidence_coverage.pdf"


def build_deletion_panel(output_dir: Path) -> Path | None:
    curve_files = sorted((output_dir / "tables").glob("*_faithfulness_curves.csv"))
    if not curve_files:
        return None
    curves = pd.read_csv(curve_files[-1])

    apply_inline_style()
    fig, ax = _panel()

    # The deletion ladder is geometric, so plot it on evenly spaced positions:
    # on a linear axis every point below k=20 collapses into the left margin.
    ladder = sorted(curves["k"].unique())
    position = {k: i for i, k in enumerate(ladder)}

    for method, (label, color, marker) in METHOD_STYLE.items():
        block = curves[curves["method"] == method]
        if block.empty:
            continue
        mean = block.groupby("k")["logit"].mean().reindex(ladder)
        std = block.groupby("k")["logit"].std().reindex(ladder)
        xs = [position[k] for k in ladder]
        ax.plot(xs, mean.values, color=color, marker=marker, markersize=2.8, linewidth=1.3, label=label, zorder=4)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.14, linewidth=0, zorder=2)

    per_node_files = sorted((output_dir / "tables").glob("*_faithfulness_per_node.csv"))
    if per_node_files:
        random_logit = float(pd.read_csv(per_node_files[-1])["logit_random_k_removed"].mean())
        ax.axhline(random_logit, color="0.35", linestyle="--", linewidth=1.1, label="random 5", zorder=3)

    # Labelled in the legend rather than annotated in place: at this panel size
    # an in-axes note cannot avoid both the curves and the legend box.
    ax.axhline(0.0, color="0.15", linestyle=(0, (1, 2.5)), linewidth=1.0,
               label="boundary", zorder=1)
    ax.set_xlabel("Features removed, $k$")
    ax.set_ylabel("Illicit logit")
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([str(k) for k in ladder], fontsize=5.8)
    ax.set_xlim(-0.35, len(ladder) - 0.65)
    ax.set_ylim(-4.9, 12.6)  # band below the curves reserved for the legend
    ax.set_yticks([0, 4, 8, 12])
    # A denser legend than the other panels: four entries must fit the same
    # axes width without spilling past the frame.
    deletion_legend = {**LEGEND_KW, "fontsize": 5.3, "handlelength": 1.25, "handletextpad": 0.35}
    ax.legend(loc="lower center", ncol=2, columnspacing=0.7, **deletion_legend)
    save_figure(fig, output_dir / "figures", "fig10b_deletion_curve", tight=False)
    return output_dir / "figures" / "fig10b_deletion_curve.pdf"


def build_revision_plots(output_dir: str | Path = "outputs") -> dict[str, str]:
    output_dir = Path(output_dir)
    builders = {
        "fig09a": build_threshold_panel,
        "fig09b": build_reliability_panel,
        "fig10a": build_coverage_panel,
        "fig10b": build_deletion_panel,
    }
    built: dict[str, str] = {}
    for name, builder in builders.items():
        path = builder(output_dir)
        if path is not None:
            built[name] = str(path)
    return built


if __name__ == "__main__":
    print(build_revision_plots())
