from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_fraud.utils.io import read_json, write_table


def _metric_files(metrics_dir: Path) -> list[Path]:
    return sorted(p for p in metrics_dir.glob("*_metrics.json") if "data_validation" not in p.name)


def collect_run_metrics(output_dir: str | Path = "outputs") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _metric_files(Path(output_dir) / "metrics"):
        data = read_json(path)
        if "metrics" not in data:
            continue
        test = data["metrics"]["test"]
        rows.append(
            {
                "run_id": data["run_id"],
                "model": data["model"],
                "seed": data["seed"],
                "feature_mode": data.get("feature_mode", "all_original_features"),
                "run_tag": data.get("run_tag", "") or "",
                "split_version": data.get("split_version", "temporal_60_20_20_v1"),
                "precision_illicit": test["precision_illicit"],
                "recall_illicit": test["recall_illicit"],
                "f1_illicit": test["f1_illicit"],
                "pr_auc": test["pr_auc"],
                "roc_auc": test["roc_auc"],
                "balanced_accuracy": test["balanced_accuracy"],
                "mcc": test["mcc"],
                "brier": test["brier"],
                "ece": test["ece"],
                "train_seconds": data["runtime"]["train_seconds"],
                "inference_seconds": data["runtime"]["inference_seconds"],
                "parameters": data.get("parameter_count", 0),
            }
        )
    return pd.DataFrame(rows)


def main_runs(output_dir: str | Path = "outputs") -> pd.DataFrame:
    """Runs that back the headline tables: full feature set, base split, no ablation tag."""
    df = collect_run_metrics(output_dir)
    if df.empty:
        return df
    keep = (
        (df["feature_mode"] == "all_original_features")
        & (df["run_tag"] == "")
        & (df["split_version"] == "temporal_60_20_20_v1")
    )
    return df.loc[keep].reset_index(drop=True)


def build_tables(output_dir: str | Path = "outputs") -> dict[str, str]:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    df = main_runs(output_dir)
    written: dict[str, str] = {}
    if not df.empty:
        write_table(df, "run_level_detection_metrics", tables_dir)
        metric_cols = [
            "precision_illicit",
            "recall_illicit",
            "f1_illicit",
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "mcc",
            "brier",
            "ece",
            "train_seconds",
            "inference_seconds",
            "parameters",
        ]
        agg = df.groupby("model")[metric_cols].agg(["mean", "std"])
        flat = agg.copy()
        flat.columns = [f"{a}_{b}" for a, b in flat.columns]
        flat = flat.reset_index()
        write_table(flat, "table03_main_detection_results", tables_dir)
        written["table03"] = str(tables_dir / "table03_main_detection_results.csv")
        eff = df.groupby("model")[["train_seconds", "inference_seconds", "parameters", "brier", "ece"]].mean().reset_index()
        write_table(eff, "table04_efficiency_calibration", tables_dir)
        written["table04"] = str(tables_dir / "table04_efficiency_calibration.csv")

    manifests = sorted((output_dir / "metrics").glob("*_split_manifest.json"))
    if manifests:
        manifest = read_json(manifests[-1])
        split_rows = []
        for split, counts in manifest["counts"].items():
            split_rows.append(
                {
                    "Split": split,
                    "Time steps": f"{min(counts['time_steps'])}-{max(counts['time_steps'])}",
                    "Total nodes": counts["total_nodes_in_period"],
                    "Labeled nodes": counts["labeled_nodes"],
                    "Illicit": counts["illicit"],
                    "Licit": counts["licit"],
                    "Unknown": counts["unknown"],
                    "Edges visible": counts["edges_visible_within_period"],
                    "Illicit ratio": counts["illicit_ratio"],
                }
            )
        split_df = pd.DataFrame(split_rows)
        write_table(split_df, "table01_dataset_split_statistics", tables_dir)
        written["table01"] = str(tables_dir / "table01_dataset_split_statistics.csv")
    return written
