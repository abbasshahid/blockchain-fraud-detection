"""Reliability diagrams and calibration repair.

Reviewer 2 asked for calibration curves rather than only the scalar ECE and
Brier score.  We produce (a) a reliability diagram over the pooled five-seed
test predictions and (b) a post-hoc Platt-recalibrated variant fitted on the
validation period only.

Platt scaling (an affine map ``a * logit(p) + b``) is used rather than pure
temperature scaling because the graph models are trained with a positive-class
weight, which shifts the decision bias upward; that offset needs the intercept
``b`` and cannot be removed by rescaling alone.  The comparison therefore
separates a fixable probability-scaling problem from a genuine ranking problem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from blockchain_fraud.analysis.plot_style import MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, apply_paper_style, save_figure
from blockchain_fraud.analysis.threshold_analysis import MODELS, SEEDS, prediction_path
from blockchain_fraud.training.evaluator import expected_calibration_error
from blockchain_fraud.utils.io import write_json, write_table

N_BINS = 10
EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """Affine logit recalibration ``a * logit(p) + b`` fitted on validation only."""
    z = _logit(p_val).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z, y_val)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def apply_platt(prob: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return _sigmoid(slope * _logit(prob) + intercept)


def reliability_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            rows.append({"bin_lower": lo, "bin_upper": hi, "count": 0, "mean_predicted": np.nan, "observed_frequency": np.nan})
            continue
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": int(mask.sum()),
                "mean_predicted": float(prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def _pooled(output_dir: Path, model: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for seed in SEEDS:
        path = prediction_path(output_dir, model, seed)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        part = df[df["split"] == split]
        ys.append(part["y_true"].to_numpy(dtype=int))
        ps.append(part["prob_illicit"].to_numpy(dtype=float))
    return np.concatenate(ys), np.concatenate(ps)


def build_calibration_artifacts(output_dir: str | Path = "outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    apply_paper_style()

    bin_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for model in MODELS:
        y_test, p_test = _pooled(output_dir, model, "test")
        y_val, p_val = _pooled(output_dir, model, "val")
        slope, intercept = fit_platt(y_val, p_val)
        p_scaled = apply_platt(p_test, slope, intercept)

        raw_bins = reliability_bins(y_test, p_test)
        raw_bins.insert(0, "variant", "raw")
        raw_bins.insert(0, "model", model)
        scaled_bins = reliability_bins(y_test, p_scaled)
        scaled_bins.insert(0, "variant", "platt_recalibrated")
        scaled_bins.insert(0, "model", model)
        bin_frames.extend([raw_bins, scaled_bins])

        summary_rows.append(
            {
                "model": model,
                "platt_slope": slope,
                "platt_intercept": intercept,
                "ece_raw": expected_calibration_error(y_test, p_test, N_BINS),
                "ece_scaled": expected_calibration_error(y_test, p_scaled, N_BINS),
                "brier_raw": float(brier_score_loss(y_test, p_test)),
                "brier_scaled": float(brier_score_loss(y_test, p_scaled)),
                "mean_predicted": float(p_test.mean()),
                "observed_base_rate": float(y_test.mean()),
            }
        )

    bins = pd.concat(bin_frames, ignore_index=True)
    write_table(bins, "calibration_reliability_bins", output_dir / "tables")
    summary = pd.DataFrame(summary_rows)
    write_table(summary, "table07_calibration", output_dir / "tables")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, variant, title in zip(axes, ["raw", "platt_recalibrated"], ["Raw probabilities", "Validation-fitted recalibration"]):
        ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1.4, label="perfect calibration")
        for model in MODELS:
            part = bins[(bins["model"] == model) & (bins["variant"] == variant)].dropna(subset=["mean_predicted"])
            ax.plot(
                part["mean_predicted"],
                part["observed_frequency"],
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
    axes[0].set_ylabel("Observed illicit frequency")
    axes[1].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig09_calibration_reliability")

    payload = {
        "bins": N_BINS,
        "pooled_seeds": SEEDS,
        "note": "Platt affine logit recalibration fitted on the validation period only, then applied unchanged to the test period.",
        "per_model": {
            r.model: {
                "platt_slope": round(float(r.platt_slope), 3),
                "platt_intercept": round(float(r.platt_intercept), 3),
                "ece_raw": round(float(r.ece_raw), 4),
                "ece_scaled": round(float(r.ece_scaled), 4),
                "brier_raw": round(float(r.brier_raw), 4),
                "brier_scaled": round(float(r.brier_scaled), 4),
                "mean_predicted": round(float(r.mean_predicted), 4),
                "observed_base_rate": round(float(r.observed_base_rate), 4),
            }
            for r in summary.itertuples(index=False)
        },
    }
    write_json(payload, output_dir / "metrics" / "calibration_summary.json")
    return {"summary": summary, "payload": payload}


if __name__ == "__main__":
    import json

    print(json.dumps(build_calibration_artifacts()["payload"]["per_model"], indent=2))
