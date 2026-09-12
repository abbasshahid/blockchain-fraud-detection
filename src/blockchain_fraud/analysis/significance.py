"""Paired significance tests over the five-seed runs.

Reviewer 2 asked whether the differences between models are statistically
meaningful.  Because every model sees exactly the same five seeds and the same
temporal split, the runs are paired, so we report both the parametric paired
t-test and the non-parametric Wilcoxon signed-rank test on the per-seed
differences.

With n = 5 paired observations the two-sided Wilcoxon signed-rank test cannot
produce a p-value below 0.0625, so we also report Cohen's d_z and the mean
difference; the Wilcoxon column should be read as a sign-consistency check
rather than as a test with useful power at alpha = 0.05.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from blockchain_fraud.analysis.make_tables import main_runs
from blockchain_fraud.utils.io import write_json, write_table

METRICS = ["f1_illicit", "pr_auc", "roc_auc", "mcc", "precision_illicit", "recall_illicit"]
MIN_WILCOXON_P = 0.0625  # two-sided floor for n = 5 pairs


def _paired_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="seed", columns="model", values=metric).dropna(axis=0, how="any")


def paired_tests(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or METRICS
    rows: list[dict[str, object]] = []
    for metric in metrics:
        wide = _paired_frame(df, metric)
        if wide.empty or wide.shape[1] < 2:
            continue
        for a, b in combinations(sorted(wide.columns), 2):
            x = wide[a].to_numpy(dtype=float)
            y = wide[b].to_numpy(dtype=float)
            diff = x - y
            n = len(diff)
            sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
            if np.allclose(diff, 0.0):
                t_p = 1.0
                w_p = 1.0
                t_stat = 0.0
            else:
                t_stat, t_p = stats.ttest_rel(x, y)
                w_p = float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
            rows.append(
                {
                    "metric": metric,
                    "model_a": a,
                    "model_b": b,
                    "n_pairs": n,
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "mean_difference": float(np.mean(diff)),
                    "cohens_dz": float(np.mean(diff) / sd) if sd > 0 else float("inf"),
                    "t_statistic": float(t_stat),
                    "t_p_value": float(t_p),
                    "wilcoxon_p_value": w_p,
                    "wilcoxon_at_floor": bool(np.isclose(w_p, MIN_WILCOXON_P)),
                    "sign_consistent": bool(np.all(diff > 0) or np.all(diff < 0)),
                }
            )
    return pd.DataFrame(rows)


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, clipped to 1.0."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def build_significance_artifacts(output_dir: str | Path = "outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    df = main_runs(output_dir)
    if df.empty:
        raise RuntimeError("No main runs found; train the headline models first.")
    tests = paired_tests(df)
    if tests.empty:
        raise RuntimeError("Not enough paired runs for significance testing.")
    for metric, block in tests.groupby("metric"):
        tests.loc[block.index, "t_p_holm"] = holm_bonferroni(block["t_p_value"].to_numpy(dtype=float))
    tests = tests.sort_values(["metric", "model_a", "model_b"]).reset_index(drop=True)
    write_table(tests, "table05_significance_tests", output_dir / "tables")

    headline = tests[(tests["metric"] == "f1_illicit")].copy()
    summary = {
        "seeds": sorted(int(s) for s in df["seed"].unique()),
        "n_pairs": int(tests["n_pairs"].max()),
        "wilcoxon_two_sided_floor": MIN_WILCOXON_P,
        "note": (
            "With five paired seeds the two-sided Wilcoxon signed-rank p-value is bounded "
            "below by 0.0625, so it reaches significance at alpha=0.10 but never at alpha=0.05."
        ),
        "f1_illicit": [
            {
                "comparison": f"{r.model_a} vs {r.model_b}",
                "mean_difference": round(float(r.mean_difference), 4),
                "t_p_value": float(f"{r.t_p_value:.3g}"),
                "t_p_holm": float(f"{r.t_p_holm:.3g}"),
                "wilcoxon_p_value": float(f"{r.wilcoxon_p_value:.3g}"),
                "cohens_dz": round(float(r.cohens_dz), 2),
            }
            for r in headline.itertuples(index=False)
        ],
    }
    write_json(summary, output_dir / "metrics" / "significance_summary.json")
    return {"tests": tests, "summary": summary}


if __name__ == "__main__":
    result = build_significance_artifacts()
    print(result["tests"].to_string(index=False))
