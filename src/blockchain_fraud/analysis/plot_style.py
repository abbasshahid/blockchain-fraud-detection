"""Shared matplotlib style for every paper figure.

Reviewer 1 asked for larger fonts and clearer legends.  All figure code goes
through :func:`apply_paper_style` so that the whole figure set is typographically
consistent and remains legible when a two-column figure is scaled to roughly
half of an LNCS text width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito derived, colour-blind safe, distinguishable in greyscale print.
MODEL_COLORS = {
    "xgboost": "#0072B2",
    "gcn": "#009E73",
    "graphsage": "#D55E00",
    "gat": "#CC79A7",
}
MODEL_MARKERS = {"xgboost": "o", "gcn": "s", "graphsage": "^", "gat": "D"}
MODEL_LABELS = {"xgboost": "XGBoost", "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT"}

BASE_FONT_SIZE = 13


def apply_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "axes.labelsize": BASE_FONT_SIZE + 1,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.7",
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, fig_dir: str | Path, stem: str, tight: bool = True) -> None:
    """Write both the PDF used by the manuscript and a PNG for quick review.

    Pass ``tight=False`` for panels whose canvas size must be preserved exactly.
    The default ``bbox_inches="tight"`` crops each figure to its own content, so
    two panels drawn on the same canvas can still come out at different sizes --
    which is precisely what makes a row of panels look ragged on the page.
    """
    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(out / f"{stem}.pdf", bbox_inches=bbox)
    fig.savefig(out / f"{stem}.png", dpi=400, bbox_inches=bbox)
    plt.close(fig)


# LNCS text width in inches (122 mm). Figures authored at this width and
# included with `width=\linewidth` are not rescaled, so the point sizes set
# here are the point sizes that reach the page.
LNCS_TEXT_WIDTH_IN = 4.8


def apply_inline_style() -> None:
    """Style for figures placed at 1:1 in the manuscript."""
    apply_paper_style()
    matplotlib.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.4,
            "grid.linewidth": 0.4,
            # Keep the full canvas: cropping to content would give each panel a
            # slightly different size and make a row of panels look ragged.
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )
