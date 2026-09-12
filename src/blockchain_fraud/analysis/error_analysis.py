"""Error analysis over the test period.

Answers three related reviewer questions with one set of statistics:

* which transactions the models get wrong, and what those transactions look
  like structurally and temporally (R2);
* why the graph models produce so many false positives compared with XGBoost
  at the fixed 0.5 threshold (R1);
* why GAT reaches the highest illicit recall but the lowest illicit F1 (R1).

The GAT question is answered by the score distribution: attention-weighted
averaging over a graph whose neighbourhoods are largely licit collapses the
logit range, so a much larger share of the test set sits above 0.5 even though
GAT's *ranking* quality (ROC-AUC) is no better than GCN's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blockchain_fraud.analysis.threshold_analysis import MODELS, SEEDS, prediction_path
from blockchain_fraud.utils.io import load_graph, write_json, write_table

PROCESSED = "data/processed/elliptic_temporal_v1_all_original_features.pt"
REFERENCE_SEED = 11


def graph_node_table(processed_path: str | Path = PROCESSED) -> pd.DataFrame:
    graph = load_graph(processed_path)
    edge_index = graph["edge_index"].numpy()
    n = len(graph["tx_id"])
    indeg = np.bincount(edge_index[1], minlength=n)
    outdeg = np.bincount(edge_index[0], minlength=n)
    return pd.DataFrame(
        {
            "tx_id": [str(t) for t in graph["tx_id"]],
            "in_degree": indeg,
            "out_degree": outdeg,
            "total_degree": indeg + outdeg,
            "time_step": graph["time_step"].numpy(),
        }
    )


def _error_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.full(len(y_true), "TN", dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    return out


def error_profiles(output_dir: str | Path = "outputs", processed_path: str | Path = PROCESSED, threshold: float = 0.5) -> pd.DataFrame:
    output_dir = Path(output_dir)
    nodes = graph_node_table(processed_path)
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            path = prediction_path(output_dir, model, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path, dtype={"tx_id": str})
            test = df[df["split"] == "test"].merge(nodes, on="tx_id", how="left", suffixes=("", "_g"))
            y_true = test["y_true"].to_numpy(dtype=int)
            prob = test["prob_illicit"].to_numpy(dtype=float)
            test = test.assign(error_class=_error_class(y_true, (prob >= threshold).astype(int)))
            for name, block in test.groupby("error_class"):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "error_class": name,
                        "n": int(len(block)),
                        "mean_prob": float(block["prob_illicit"].mean()),
                        "mean_total_degree": float(block["total_degree"].mean()),
                        "median_total_degree": float(block["total_degree"].median()),
                        "mean_in_degree": float(block["in_degree"].mean()),
                        "mean_out_degree": float(block["out_degree"].mean()),
                        "degree_one_share": float((block["total_degree"] <= 1).mean()),
                        "mean_time_step": float(block["time_step"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def per_timestep_performance(output_dir: str | Path = "outputs", threshold: float = 0.5) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            path = prediction_path(output_dir, model, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path, dtype={"tx_id": str})
            test = df[df["split"] == "test"]
            for step, block in test.groupby("time_step"):
                y = block["y_true"].to_numpy(dtype=int)
                pred = (block["prob_illicit"].to_numpy(dtype=float) >= threshold).astype(int)
                tp = int(((y == 1) & (pred == 1)).sum())
                fp = int(((y == 0) & (pred == 1)).sum())
                fn = int(((y == 1) & (pred == 0)).sum())
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "time_step": int(step),
                        "illicit": int((y == 1).sum()),
                        "labelled": int(len(y)),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
                    }
                )
    return pd.DataFrame(rows)


def score_distribution(output_dir: str | Path = "outputs") -> pd.DataFrame:
    """Where each model's test scores sit relative to the fixed 0.5 threshold."""
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            path = prediction_path(output_dir, model, seed)
            if not path.exists():
                continue
            test = pd.read_csv(path).query("split == 'test'")
            prob = test["prob_illicit"].to_numpy(dtype=float)
            y = test["y_true"].to_numpy(dtype=int)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "share_above_0.5": float((prob >= 0.5).mean()),
                    "median_prob_licit": float(np.median(prob[y == 0])),
                    "median_prob_illicit": float(np.median(prob[y == 1])),
                    "iqr_prob_licit": float(np.subtract(*np.percentile(prob[y == 0], [75, 25]))),
                    "score_separation": float(np.median(prob[y == 1]) - np.median(prob[y == 0])),
                    "interquartile_range": float(np.subtract(*np.percentile(prob, [75, 25]))),
                }
            )
    return pd.DataFrame(rows)


def model_agreement(output_dir: str | Path = "outputs", seed: int = REFERENCE_SEED) -> dict[str, object]:
    output_dir = Path(output_dir)
    frames = {}
    for model in MODELS:
        path = prediction_path(output_dir, model, seed)
        if path.exists():
            frames[model] = pd.read_csv(path, dtype={"tx_id": str}).query("split == 'test'").set_index("tx_id")
    if "xgboost" not in frames or "graphsage" not in frames:
        return {}
    xgb = frames["xgboost"]
    sage = frames["graphsage"].reindex(xgb.index)
    y = xgb["y_true"].to_numpy(dtype=int)
    px = (xgb["prob_illicit"].to_numpy(dtype=float) >= 0.5).astype(int)
    ps = (sage["prob_illicit"].to_numpy(dtype=float) >= 0.5).astype(int)
    illicit = y == 1
    return {
        "seed": seed,
        "test_illicit": int(illicit.sum()),
        "caught_by_both": int((illicit & (px == 1) & (ps == 1)).sum()),
        "caught_only_by_xgboost": int((illicit & (px == 1) & (ps == 0)).sum()),
        "caught_only_by_graphsage": int((illicit & (px == 0) & (ps == 1)).sum()),
        "missed_by_both": int((illicit & (px == 0) & (ps == 0)).sum()),
        "union_recall": float((illicit & ((px == 1) | (ps == 1))).sum() / max(1, illicit.sum())),
        "intersection_alerts": int(((px == 1) & (ps == 1)).sum()),
        "intersection_precision": float(y[(px == 1) & (ps == 1)].mean()) if ((px == 1) & (ps == 1)).any() else 0.0,
        "xgboost_alerts": int((px == 1).sum()),
        "graphsage_alerts": int((ps == 1).sum()),
    }


def build_error_artifacts(output_dir: str | Path = "outputs", processed_path: str | Path = PROCESSED) -> dict[str, object]:
    output_dir = Path(output_dir)
    profiles = error_profiles(output_dir, processed_path)
    write_table(profiles, "error_profiles_raw", output_dir / "tables")
    mean_profiles = (
        profiles.groupby(["model", "error_class"])[
            ["n", "mean_prob", "mean_total_degree", "median_total_degree", "degree_one_share", "mean_time_step"]
        ]
        .mean()
        .reset_index()
    )
    write_table(mean_profiles, "table08_error_analysis", output_dir / "tables")

    steps = per_timestep_performance(output_dir)
    write_table(steps, "error_per_timestep_raw", output_dir / "tables")
    step_mean = steps.groupby(["model", "time_step"])[["illicit", "tp", "fn", "recall", "f1"]].mean().reset_index()
    write_table(step_mean, "error_per_timestep_mean", output_dir / "tables")

    scores = score_distribution(output_dir)
    write_table(scores, "error_score_distribution_raw", output_dir / "tables")
    score_mean = scores.groupby("model").mean(numeric_only=True).drop(columns=["seed"]).reset_index()
    write_table(score_mean, "table09_score_distribution", output_dir / "tables")

    agreement = model_agreement(output_dir)

    payload: dict[str, object] = {
        "threshold": 0.5,
        "seeds": SEEDS,
        "error_profiles": {
            model: {
                row.error_class: {
                    "n": round(float(row.n), 1),
                    "mean_total_degree": round(float(row.mean_total_degree), 2),
                    "median_total_degree": round(float(row.median_total_degree), 2),
                    "degree_one_share": round(float(row.degree_one_share), 3),
                    "mean_time_step": round(float(row.mean_time_step), 2),
                }
                for row in block.itertuples(index=False)
            }
            for model, block in mean_profiles.groupby("model")
        },
        "model_agreement_seed11": agreement,
    }
    payload["score_distribution"] = {
        r["model"]: {
            "share_above_0.5": round(float(r["share_above_0.5"]), 4),
            "median_prob_licit": round(float(r["median_prob_licit"]), 4),
            "median_prob_illicit": round(float(r["median_prob_illicit"]), 4),
            "score_separation": round(float(r["score_separation"]), 4),
            "interquartile_range": round(float(r["interquartile_range"]), 4),
        }
        for _, r in score_mean.iterrows()
    }
    worst_steps = (
        step_mean[step_mean["model"] == "xgboost"].sort_values("recall").head(3)[["time_step", "illicit", "recall"]].to_dict("records")
    )
    payload["xgboost_worst_test_steps"] = worst_steps
    write_json(payload, output_dir / "metrics" / "error_analysis_summary.json")
    return {"profiles": mean_profiles, "steps": step_mean, "scores": score_mean, "payload": payload}


if __name__ == "__main__":
    import json

    result = build_error_artifacts()
    print(json.dumps(result["payload"]["score_distribution"], indent=2))
    print(json.dumps(result["payload"]["model_agreement_seed11"], indent=2))
    print(result["profiles"].to_string(index=False))
