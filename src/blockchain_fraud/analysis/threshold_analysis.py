"""Decision-threshold sensitivity and alert-budget analysis.

The headline table fixes tau = 0.5, which is what makes the graph models look
like high-recall, low-precision detectors.  Reviewer 2 asked how the framework
behaves under other operating points, and Reviewer 1 asked why precision is so
low.  Both are answered by the same three views:

1. a full sweep of tau over the test split (precision / recall / F1 / alert rate);
2. a validation-tuned threshold tau*, selected on the validation period only and
   then applied once to the test period, which is the leakage-free way an
   operator would pick an operating point;
3. an alert-budget view (precision@k), which measures how many of the top-k
   ranked transactions are truly illicit and is the quantity an analyst team
   with finite review capacity actually cares about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blockchain_fraud.analysis.make_tables import main_runs
from blockchain_fraud.utils.io import write_json, write_table

MODELS = ["xgboost", "graphsage", "gcn", "gat"]
SEEDS = [11, 22, 33, 44, 55]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
BUDGETS = [100, 250, 500, 1000]


def prediction_path(output_dir: Path, model: str, seed: int) -> Path:
    return output_dir / "predictions" / f"elliptic_temporal_v1_{model}_all_original_features_seed{seed}_predictions.csv"


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def _prf(y_true: np.ndarray, prob: np.ndarray, tau: float) -> dict[str, float]:
    tp, fp, fn, tn = _counts(y_true, (prob >= tau).astype(int))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / max(1, len(y_true)),
    }


def _precision_at_k(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    k = min(k, len(prob))
    top = np.argsort(-prob, kind="stable")[:k]
    return float(y_true[top].mean())


def sweep(output_dir: str | Path = "outputs", models: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    models = models or MODELS
    seeds = seeds or SEEDS
    rows: list[dict[str, object]] = []
    for model in models:
        for seed in seeds:
            path = prediction_path(output_dir, model, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            test = df[df["split"] == "test"]
            y_true = test["y_true"].to_numpy(dtype=int)
            prob = test["prob_illicit"].to_numpy(dtype=float)
            for tau in THRESHOLDS:
                rows.append({"model": model, "seed": seed, "threshold": float(tau), **_prf(y_true, prob, float(tau))})
    return pd.DataFrame(rows)


def tuned_operating_points(output_dir: str | Path = "outputs", models: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    """Pick tau on the validation period, then evaluate once on the test period."""
    output_dir = Path(output_dir)
    models = models or MODELS
    seeds = seeds or SEEDS
    rows: list[dict[str, object]] = []
    for model in models:
        for seed in seeds:
            path = prediction_path(output_dir, model, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            val = df[df["split"] == "val"]
            test = df[df["split"] == "test"]
            y_val = val["y_true"].to_numpy(dtype=int)
            p_val = val["prob_illicit"].to_numpy(dtype=float)
            y_test = test["y_true"].to_numpy(dtype=int)
            p_test = test["prob_illicit"].to_numpy(dtype=float)
            best_tau, best_f1 = 0.5, -1.0
            for tau in THRESHOLDS:
                f1 = _prf(y_val, p_val, float(tau))["f1"]
                if f1 > best_f1:
                    best_f1, best_tau = f1, float(tau)
            fixed = _prf(y_test, p_test, 0.5)
            tuned = _prf(y_test, p_test, best_tau)
            row: dict[str, object] = {
                "model": model,
                "seed": seed,
                "tau_star": best_tau,
                "val_f1_at_tau_star": best_f1,
                "test_f1_at_0.5": fixed["f1"],
                "test_precision_at_0.5": fixed["precision"],
                "test_recall_at_0.5": fixed["recall"],
                "test_fp_at_0.5": fixed["fp"],
                "test_f1_at_tau_star": tuned["f1"],
                "test_precision_at_tau_star": tuned["precision"],
                "test_recall_at_tau_star": tuned["recall"],
                "test_fp_at_tau_star": tuned["fp"],
            }
            for k in BUDGETS:
                row[f"precision_at_{k}"] = _precision_at_k(y_test, p_test, k)
            rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_artifacts(output_dir: str | Path = "outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    full = sweep(output_dir)
    if full.empty:
        raise RuntimeError("No prediction files found for the threshold sweep.")
    write_table(full, "threshold_sweep_raw", output_dir / "tables")

    mean_sweep = full.groupby(["model", "threshold"])[["precision", "recall", "f1", "fp", "alert_rate"]].mean().reset_index()
    write_table(mean_sweep, "threshold_sweep_mean", output_dir / "tables")

    tuned = tuned_operating_points(output_dir)
    write_table(tuned, "threshold_tuned_raw", output_dir / "tables")

    agg_cols = [c for c in tuned.columns if c not in {"model", "seed"}]
    summary = tuned.groupby("model")[agg_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    write_table(summary, "table06_threshold_sensitivity", output_dir / "tables")

    best_f1 = (
        mean_sweep.loc[mean_sweep.groupby("model")["f1"].idxmax()][["model", "threshold", "precision", "recall", "f1", "fp"]]
        .rename(columns={"threshold": "oracle_test_threshold"})
        .reset_index(drop=True)
    )
    write_table(best_f1, "threshold_oracle_best_f1", output_dir / "tables")

    reference = main_runs(output_dir)
    payload = {
        "thresholds_evaluated": [float(t) for t in THRESHOLDS],
        "alert_budgets": BUDGETS,
        "validation_tuned": {
            row["model"]: {
                "tau_star_mean": round(float(row["tau_star_mean"]), 3),
                "test_f1_at_0.5": round(float(row["test_f1_at_0.5_mean"]), 4),
                "test_f1_at_tau_star": round(float(row["test_f1_at_tau_star_mean"]), 4),
                "test_precision_at_tau_star": round(float(row["test_precision_at_tau_star_mean"]), 4),
                "test_recall_at_tau_star": round(float(row["test_recall_at_tau_star_mean"]), 4),
                "false_positives_at_0.5": round(float(row["test_fp_at_0.5_mean"]), 1),
                "false_positives_at_tau_star": round(float(row["test_fp_at_tau_star_mean"]), 1),
                **{f"precision_at_{k}": round(float(row[f"precision_at_{k}_mean"]), 4) for k in BUDGETS},
            }
            for _, row in summary.iterrows()
        },
        "oracle_best_f1_on_test": {
            r.model: {"threshold": float(r.oracle_test_threshold), "f1": round(float(r.f1), 4)}
            for r in best_f1.itertuples(index=False)
        },
        "reference_f1_at_0.5": {
            m: round(float(v), 4) for m, v in reference.groupby("model")["f1_illicit"].mean().items()
        },
        "test_illicit_count": int(
            pd.read_csv(prediction_path(output_dir, "xgboost", 11)).query("split == 'test'")["y_true"].sum()
        ),
    }
    write_json(payload, output_dir / "metrics" / "threshold_sensitivity_summary.json")
    return {"sweep": mean_sweep, "tuned": summary, "summary": payload}


if __name__ == "__main__":
    out = build_threshold_artifacts()
    import json

    print(json.dumps(out["summary"]["validation_tuned"], indent=2))
