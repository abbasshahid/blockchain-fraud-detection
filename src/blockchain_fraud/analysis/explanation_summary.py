"""Aggregate every generated-report validation run into one table.

Reviewer 1 asked for the explanation evaluation to be strengthened or its
conclusions moderated, and for the evidence-coverage metric to be defined.
This module collects all validation files, reports the sample size actually
achieved per provider and method, and tests the grounded-vs-unconstrained
coverage difference on the transactions the two runs share, using a paired
Wilcoxon signed-rank test rather than a bare difference of means.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from blockchain_fraud.utils.io import read_json, write_json, write_table

PATTERN = re.compile(r"_(?P<method>template|grounded_llm|unconstrained_llm)(?:_(?P<provider>[a-z0-9]+))?_report_validation\.json$")


def _label(method: str, provider: str | None, version: str) -> str:
    provider_label = {"openrouter": "OpenRouter", None: "Gemini", "": "Gemini"}.get(provider, str(provider).title())
    method_label = {"grounded_llm": "grounded", "unconstrained_llm": "unconstrained", "template": "template"}[method]
    return f"{provider_label} {method_label} ({version})"


def collect_validations(output_dir: str | Path = "outputs") -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "metrics").glob("*_report_validation.json")):
        match = PATTERN.search(path.name)
        if not match:
            continue
        records = read_json(path)
        if not isinstance(records, list) or not records:
            continue
        version = "v2" if "_btrep_v2_" in path.name else "v1"
        for record in records:
            rows.append(
                {
                    "source_file": path.name,
                    "method": match.group("method"),
                    "provider": match.group("provider") or "gemini",
                    "evidence_version": record.get("evidence_version", version),
                    "label": _label(match.group("method"), match.group("provider"), version),
                    "transaction_id": str(record.get("transaction_id")),
                    "valid": bool(record.get("valid", False)),
                    "citation_validity": float(record.get("citation_validity", np.nan)),
                    "evidence_coverage": float(record.get("evidence_coverage", np.nan)),
                    "available_id_count": record.get("available_id_count"),
                    "unsupported_claim_count": record.get("unsupported_claim_count", 0),
                    "reason_count": record.get("reason_count", 0),
                }
            )
    return pd.DataFrame(rows)


def paired_coverage_test(df: pd.DataFrame, version: str = "v2", provider: str = "gemini") -> dict[str, object]:
    grounded = df[(df["method"] == "grounded_llm") & (df["evidence_version"] == version) & (df["provider"] == provider)]
    unconstrained = df[(df["method"] == "unconstrained_llm") & (df["evidence_version"] == version) & (df["provider"] == provider)]
    if grounded.empty or unconstrained.empty:
        return {}
    merged = grounded[["transaction_id", "evidence_coverage"]].merge(
        unconstrained[["transaction_id", "evidence_coverage"]], on="transaction_id", suffixes=("_grounded", "_unconstrained")
    )
    if len(merged) < 3:
        return {"n_pairs": int(len(merged))}
    diff = merged["evidence_coverage_grounded"] - merged["evidence_coverage_unconstrained"]
    result: dict[str, object] = {
        "n_pairs": int(len(merged)),
        "mean_grounded": round(float(merged["evidence_coverage_grounded"].mean()), 4),
        "mean_unconstrained": round(float(merged["evidence_coverage_unconstrained"].mean()), 4),
        "mean_difference": round(float(diff.mean()), 4),
    }
    if not np.allclose(diff, 0.0):
        test = stats.wilcoxon(
            merged["evidence_coverage_grounded"], merged["evidence_coverage_unconstrained"], zero_method="wilcox"
        )
        result["wilcoxon_statistic"] = float(test.statistic)
        result["wilcoxon_p_value"] = float(f"{test.pvalue:.3g}")
        sd = float(diff.std(ddof=1))
        result["cohens_dz"] = round(float(diff.mean() / sd), 3) if sd > 0 else None
    else:
        result["wilcoxon_p_value"] = 1.0
    return result


def build_explanation_summary(output_dir: str | Path = "outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    df = collect_validations(output_dir)
    if df.empty:
        raise RuntimeError("No report validation files found.")
    write_table(df, "explanation_validation_raw", output_dir / "tables")

    grouped = (
        df.groupby(["label", "method", "provider", "evidence_version"])
        .agg(
            reports=("transaction_id", "count"),
            valid=("valid", "sum"),
            valid_rate=("valid", "mean"),
            citation_validity=("citation_validity", "mean"),
            evidence_coverage=("evidence_coverage", "mean"),
            evidence_coverage_std=("evidence_coverage", "std"),
            unsupported_claims=("unsupported_claim_count", "mean"),
            available_groups=("available_id_count", "mean"),
        )
        .reset_index()
        .sort_values(["evidence_version", "method", "provider"])
    )
    write_table(grouped, "table17_explanation_validation", output_dir / "tables")

    payload = {
        "methods": [
            {
                "label": row["label"],
                "method": row["method"],
                "provider": row["provider"],
                "evidence_version": row["evidence_version"],
                "reports": int(row["reports"]),
                "valid": int(row["valid"]),
                "valid_rate": round(float(row["valid_rate"]), 4),
                "citation_validity": round(float(row["citation_validity"]), 4),
                "evidence_coverage": None if pd.isna(row["evidence_coverage"]) else round(float(row["evidence_coverage"]), 4),
                "evidence_coverage_std": None
                if pd.isna(row["evidence_coverage_std"])
                else round(float(row["evidence_coverage_std"]), 4),
                "unsupported_claims_per_report": round(float(row["unsupported_claims"]), 4),
                "available_evidence_groups": None
                if pd.isna(row["available_groups"])
                else round(float(row["available_groups"]), 2),
            }
            for _, row in grouped.iterrows()
        ],
        "paired_tests": {
            f"{provider}_{version}": paired_coverage_test(df, version, provider)
            for provider in sorted(df["provider"].unique())
            for version in sorted(df["evidence_version"].unique())
            if paired_coverage_test(df, version, provider)
        },
        "coverage_definition": "|cited ids that exist in the package| / |ids supplied in the package|",
    }
    write_json(payload, output_dir / "metrics" / "explanation_summary.json")
    return payload


if __name__ == "__main__":
    import json

    print(json.dumps(build_explanation_summary(), indent=2))
