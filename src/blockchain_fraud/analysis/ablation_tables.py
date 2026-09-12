"""Summary tables for the three retraining ablations.

* ``local_only``   -- 93 local Elliptic features vs the full 165-dimensional
  vector, which separates the value of the dataset's pre-aggregated
  neighbourhood statistics from the value of learned message passing (R1).
* ``splits``       -- 50/20/30 and 70/15/15 temporal boundaries alongside the
  60/20/20 protocol used in the main table, to check that the model ordering is
  a property of the data and not of one particular cut point (R2).
* ``nounknown``    -- unknown-labelled transactions removed from message
  passing, quantifying how much the unlabelled context contributes (R2).

Each table is accompanied by paired significance tests across the five shared
seeds, so the deltas are reported with the same rigour as the main results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from blockchain_fraud.analysis.make_tables import collect_run_metrics
from blockchain_fraud.utils.io import write_json, write_table

MODEL_ORDER = ["xgboost", "graphsage", "gcn", "gat"]
METRICS = ["f1_illicit", "pr_auc", "roc_auc", "precision_illicit", "recall_illicit"]
BASE_SPLIT = "temporal_60_20_20_v1"


def _paired_delta(treatment: pd.DataFrame, control: pd.DataFrame, metric: str) -> dict[str, float]:
    merged = treatment[["seed", metric]].merge(control[["seed", metric]], on="seed", suffixes=("_t", "_c"))
    if merged.empty:
        return {"delta": float("nan"), "p_value": float("nan"), "n_pairs": 0}
    diff = merged[f"{metric}_t"].to_numpy(dtype=float) - merged[f"{metric}_c"].to_numpy(dtype=float)
    if len(diff) < 2 or np.allclose(diff, 0.0):
        p = 1.0
    else:
        p = float(stats.ttest_rel(merged[f"{metric}_t"], merged[f"{metric}_c"]).pvalue)
    return {"delta": float(diff.mean()), "p_value": p, "n_pairs": int(len(diff))}


def _select(df: pd.DataFrame, feature_mode: str, run_tag: str, split_version: str) -> pd.DataFrame:
    return df[
        (df["feature_mode"] == feature_mode) & (df["run_tag"] == run_tag) & (df["split_version"] == split_version)
    ]


def local_only_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        full = _select(df, "all_original_features", "", BASE_SPLIT).query("model == @model")
        local = _select(df, "local_only", "", BASE_SPLIT).query("model == @model")
        if full.empty or local.empty:
            continue
        row: dict[str, object] = {"model": model, "n_seeds": int(min(len(full), len(local)))}
        for metric in ["f1_illicit", "pr_auc"]:
            test = _paired_delta(local, full, metric)
            row[f"{metric}_all"] = float(full[metric].mean())
            row[f"{metric}_local"] = float(local[metric].mean())
            row[f"{metric}_delta"] = test["delta"]
            row[f"{metric}_p"] = test["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def split_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_version, label in [
        (BASE_SPLIT, "60/20/20"),
        ("temporal_50_20_30_v1", "50/20/30"),
        ("temporal_70_15_15_v1", "70/15/15"),
    ]:
        block = _select(df, "all_original_features", "", split_version)
        if block.empty:
            continue
        for model in MODEL_ORDER:
            part = block.query("model == @model")
            if part.empty:
                continue
            rows.append(
                {
                    "split": label,
                    "split_version": split_version,
                    "model": model,
                    "n_seeds": int(len(part)),
                    "f1_illicit_mean": float(part["f1_illicit"].mean()),
                    "f1_illicit_std": float(part["f1_illicit"].std(ddof=1)) if len(part) > 1 else 0.0,
                    "pr_auc_mean": float(part["pr_auc"].mean()),
                    "pr_auc_std": float(part["pr_auc"].std(ddof=1)) if len(part) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def rank_stability(split_table: pd.DataFrame) -> dict[str, object]:
    """Do the alternative boundaries preserve the model ordering?"""
    if split_table.empty:
        return {}
    pivot = split_table.pivot(index="model", columns="split", values="f1_illicit_mean").dropna(axis=1, how="any")
    if pivot.shape[1] < 2:
        return {}
    ranks = pivot.rank(ascending=False)
    base = "60/20/20"
    out: dict[str, object] = {
        "ordering_by_split": {col: list(pivot[col].sort_values(ascending=False).index) for col in pivot.columns},
        "identical_ordering_everywhere": bool(
            len({tuple(pivot[col].sort_values(ascending=False).index) for col in pivot.columns}) == 1
        ),
    }
    if base in ranks.columns:
        out["spearman_vs_base"] = {
            col: round(float(stats.spearmanr(ranks[base], ranks[col]).statistic), 4)
            for col in ranks.columns
            if col != base
        }
    return out


def unknown_node_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["graphsage", "gcn", "gat"]:
        retained = _select(df, "all_original_features", "", BASE_SPLIT).query("model == @model")
        removed = _select(df, "all_original_features", "nounknown", BASE_SPLIT).query("model == @model")
        if retained.empty or removed.empty:
            continue
        row: dict[str, object] = {"model": model, "n_seeds": int(min(len(retained), len(removed)))}
        for metric in ["f1_illicit", "pr_auc", "recall_illicit", "precision_illicit"]:
            test = _paired_delta(removed, retained, metric)
            row[f"{metric}_retained"] = float(retained[metric].mean())
            row[f"{metric}_removed"] = float(removed[metric].mean())
            row[f"{metric}_delta"] = test["delta"]
            row[f"{metric}_p"] = test["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_tables(output_dir: str | Path = "outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    df = collect_run_metrics(output_dir)
    if df.empty:
        raise RuntimeError("No run metrics found.")

    local = local_only_table(df)
    splits = split_robustness_table(df)
    unknown = unknown_node_table(df)

    if not local.empty:
        write_table(local, "table12_local_only_ablation", output_dir / "tables")
    if not splits.empty:
        write_table(splits, "table13_split_robustness", output_dir / "tables")
    if not unknown.empty:
        write_table(unknown, "table14_unknown_node_ablation", output_dir / "tables")

    payload: dict[str, object] = {
        "local_only": {
            r.model: {
                "f1_all": round(float(r.f1_illicit_all), 4),
                "f1_local": round(float(r.f1_illicit_local), 4),
                "f1_delta": round(float(r.f1_illicit_delta), 4),
                "f1_p": float(f"{r.f1_illicit_p:.3g}"),
                "pr_auc_all": round(float(r.pr_auc_all), 4),
                "pr_auc_local": round(float(r.pr_auc_local), 4),
                "pr_auc_delta": round(float(r.pr_auc_delta), 4),
            }
            for r in local.itertuples(index=False)
        }
        if not local.empty
        else {},
        "split_robustness": {
            f"{r.split}|{r.model}": {
                "f1": round(float(r.f1_illicit_mean), 4),
                "f1_std": round(float(r.f1_illicit_std), 4),
                "pr_auc": round(float(r.pr_auc_mean), 4),
            }
            for r in splits.itertuples(index=False)
        }
        if not splits.empty
        else {},
        "rank_stability": rank_stability(splits),
        "unknown_nodes": {
            r.model: {
                "f1_retained": round(float(r.f1_illicit_retained), 4),
                "f1_removed": round(float(r.f1_illicit_removed), 4),
                "f1_delta": round(float(r.f1_illicit_delta), 4),
                "f1_p": float(f"{r.f1_illicit_p:.3g}"),
                "pr_auc_retained": round(float(r.pr_auc_retained), 4),
                "pr_auc_removed": round(float(r.pr_auc_removed), 4),
                "pr_auc_delta": round(float(r.pr_auc_delta), 4),
                "pr_auc_p": float(f"{r.pr_auc_p:.3g}"),
            }
            for r in unknown.itertuples(index=False)
        }
        if not unknown.empty
        else {},
    }
    write_json(payload, output_dir / "metrics" / "ablation_summary.json")
    return payload


if __name__ == "__main__":
    import json

    print(json.dumps(build_ablation_tables(), indent=2))
