"""Ranking, confusion, training and explanation figures for the manuscript.

All figures go through :mod:`blockchain_fraud.analysis.plot_style`, which sets
a larger base font, thicker lines, explicit markers and framed legends.  The
earlier versions used matplotlib defaults, which were criticised as hard to
read once a figure is scaled down to roughly half of an LNCS text width.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from blockchain_fraud.analysis.plot_style import MODEL_COLORS, MODEL_LABELS, apply_paper_style, save_figure

MODELS = ["xgboost", "gcn", "graphsage", "gat"]
LINESTYLES = {"xgboost": "-", "graphsage": "--", "gcn": "-.", "gat": ":"}


def _prediction_path(output_dir: Path, model: str, seed: int = 11) -> Path:
    return output_dir / "predictions" / f"elliptic_temporal_v1_{model}_all_original_features_seed{seed}_predictions.csv"


def _test_predictions(output_dir: Path, model: str, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(_prediction_path(output_dir, model, seed))
    test = df[df["split"] == "test"]
    return test["y_true"].to_numpy(dtype=int), test["prob_illicit"].to_numpy(dtype=float)


def build_extra_paper_plots(output_dir: str | Path = "outputs", seed: int = 11) -> None:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    # -- precision-recall ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    pr_rows = []
    for model in MODELS:
        y, p = _test_predictions(output_dir, model, seed)
        precision, recall, _ = precision_recall_curve(y, p)
        ap = float(average_precision_score(y, p))
        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS[model],
            linestyle=LINESTYLES[model],
            label=f"{MODEL_LABELS[model]} (AP = {ap:.3f})",
        )
        pr_rows.append({"model": model, "average_precision": ap})
    base_rate = float(y.mean())
    ax.axhline(base_rate, color="0.45", linestyle=(0, (1, 3)), linewidth=1.4, label=f"illicit base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig04_precision_recall_curves")
    pd.DataFrame(pr_rows).to_csv(fig_dir / "fig04_precision_recall_curves_data.csv", index=False)

    # -- ROC ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    roc_rows = []
    for model in MODELS:
        y, p = _test_predictions(output_dir, model, seed)
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model],
            linestyle=LINESTYLES[model],
            label=f"{MODEL_LABELS[model]} (AUC = {auc:.3f})",
        )
        roc_rows.append({"model": model, "roc_auc": auc})
    ax.plot([0, 1], [0, 1], color="0.45", linestyle=(0, (1, 3)), linewidth=1.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig05_roc_curves")
    pd.DataFrame(roc_rows).to_csv(fig_dir / "fig05_roc_curves_data.csv", index=False)

    # -- confusion matrices -----------------------------------------------
    for model in ["xgboost", "graphsage"]:
        metrics_path = output_dir / "metrics" / f"elliptic_temporal_v1_{model}_all_original_features_seed{seed}_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        cm = np.asarray(metrics["metrics"]["test"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.1, 3.6))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["licit", "illicit"])
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        for text in display.text_.ravel():
            text.set_fontsize(14)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.grid(False)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig06_confusion_{model}")

    # -- training curve ----------------------------------------------------
    metrics_path = output_dir / "metrics" / f"elliptic_temporal_v1_graphsage_all_original_features_seed{seed}_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    hist = pd.DataFrame(metrics["history"])
    hist.to_csv(fig_dir / "fig07_graphsage_training_curve_data.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
    ax1.plot(hist["epoch"], hist["loss"], color=MODEL_COLORS["xgboost"], label="training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted BCE loss")
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(hist["epoch"], hist["val_pr_auc"], color=MODEL_COLORS["graphsage"], linestyle="--", label="validation PR-AUC")
    ax2.set_ylabel("Validation PR-AUC")
    best_epoch = int(metrics.get("best_epoch", 0))
    if best_epoch:
        ax1.axvline(best_epoch, color="0.45", linestyle=(0, (1, 3)), linewidth=1.4)
        ax1.annotate(
            f"selected epoch {best_epoch}",
            xy=(best_epoch, ax1.get_ylim()[1]),
            xytext=(-6, -14),
            textcoords="offset points",
            ha="right",
            fontsize=11,
            color="0.25",
        )
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper center")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig07_graphsage_training_curve")


if __name__ == "__main__":
    build_extra_paper_plots()
