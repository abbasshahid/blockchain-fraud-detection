"""How much does each BTREP evidence component contribute?

Reviewer 2 asked us to justify the composition of the profile rather than
assert it.  The question is answered on the detector, not on the generated
prose, so that the answer does not depend on an API quota or on a particular
language model.

Each observational evidence group is turned into its numeric fields, and we
ask how well the groups jointly track the quantity the profile exists to
explain: the detector's illicit logit for that transaction.  Gradient-boosted
regression is fitted on a random half of the profiled transactions and scored
on the other half, and each group is then removed in turn.  The drop in
out-of-sample R^2 is that group's contribution; a permutation-importance
figure is reported alongside it as a second, model-agnostic reading.

Two deliberate choices:

* The sample is **stratified across risk deciles**.  On the top-risk sample
  used for investigation reports the logit is nearly constant, so no group
  could be shown to track it and every ablation would read as null.
* The **counterfactual** group is excluded from the pool.  It is derived from
  the detector's own logit under intervention, so including it would be close
  to tautological.  Its value is established separately, by the faithfulness
  measurements and by how often the report generator actually cites it.
* For the same reason the **feature-salience** group is scored using the
  model-agnostic magnitude proxy rather than the gradient attribution that
  BTREP v2 ships.  Gradient x input is computed from the detector, so scoring
  it against the detector's own logit would inflate its apparent contribution.
  The proxy ranks the same features by an observational criterion, which keeps
  the comparison between groups honest; the gradient variant's advantage is
  established separately, by the deletion curves.

Run this on a profile built *without* a checkpoint, so that the salience field
holds the proxy:

    build_btrep(..., selection="stratified", model_name=None)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from blockchain_fraud.utils.io import read_json, write_json, write_table

MOTIF_LABELS = ["fan_in_like", "fan_out_like", "chain_like", "isolated_or_local", "dense_neighborhood_like"]
OBSERVATIONAL_GROUPS = ["structural", "temporal", "risk_exposure", "motifs", "feature_salience"]
EPS = 1e-7
RANDOM_STATE = 11


def _logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1 - EPS))
    return float(np.log(p / (1 - p)))


def group_features(item: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Numeric fields of each observational evidence group, keyed by group."""
    ev = item["evidence"]
    out: dict[str, dict[str, float]] = {}

    struct = ev.get("structural", {})
    out["structural"] = {
        f"structural__{k}": float(struct[k])
        for k in ["in_degree", "out_degree", "total_degree", "one_hop_size", "two_hop_size_capped", "component_size"]
        if k in struct
    }

    temporal = ev.get("temporal", {})
    out["temporal"] = {
        f"temporal__{k}": float(temporal[k])
        for k in ["relative_time", "time_step", "degree_train_percentile_proxy"]
        if k in temporal
    }

    risk = ev.get("risk_exposure", {})
    out["risk_exposure"] = {
        f"risk_exposure__{k}": float(risk[k])
        for k in [
            "mean_neighbor_risk",
            "max_neighbor_risk",
            "high_risk_neighbor_count",
            "training_history_labeled_neighbors",
            "training_history_illicit_neighbors",
        ]
        if k in risk
    }

    motifs = set(ev.get("motifs", {}).get("values", []))
    out["motifs"] = {f"motifs__{label}": float(label in motifs) for label in MOTIF_LABELS}

    salience = ev.get("feature_salience") or ev.get("feature_salience_proxy") or {}
    values = salience.get("top_features") or salience.get("top_standardized_features") or []
    out["feature_salience"] = {
        f"feature_salience__rank{i}": float(entry.get("standardized_value", 0.0)) for i, entry in enumerate(values)
    }
    return out


def build_design_matrix(items: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    columns: dict[str, list[str]] = {}
    for item in items:
        groups = group_features(item)
        row: dict[str, float] = {}
        for name, fields in groups.items():
            columns.setdefault(name, list(fields.keys()))
            row.update(fields)
        rows.append(row)
        targets.append(_logit(float(item["calibrated_probability"])))
    frame = pd.DataFrame(rows).fillna(0.0)
    for name, names in columns.items():
        columns[name] = [c for c in names if c in frame.columns]
    return frame, np.asarray(targets, dtype=float), columns


def _fit_score(x_train, y_train, x_test, y_test) -> float:
    if x_train.shape[1] == 0:
        return float(r2_score(y_test, np.full_like(y_test, float(np.mean(y_train)))))
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=300, learning_rate=0.06)
    model.fit(x_train, y_train)
    return float(r2_score(y_test, model.predict(x_test)))


def evidence_value(items: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, target, columns = build_design_matrix(items)
    x_train, x_test, y_train, y_test = train_test_split(frame, target, test_size=0.4, random_state=RANDOM_STATE)
    full = _fit_score(x_train, y_train, x_test, y_test)

    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=300, learning_rate=0.06)
    model.fit(x_train, y_train)
    perm = permutation_importance(model, x_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="r2")
    perm_by_column = dict(zip(frame.columns, perm.importances_mean))

    rows: list[dict[str, Any]] = []
    for group in OBSERVATIONAL_GROUPS:
        cols = columns.get(group, [])
        if not cols:
            continue
        without = [c for c in frame.columns if c not in cols]
        only = cols
        rows.append(
            {
                "group": group,
                "fields": len(cols),
                "r2_full": full,
                "r2_without_group": _fit_score(x_train[without], y_train, x_test[without], y_test),
                "r2_group_alone": _fit_score(x_train[only], y_train, x_test[only], y_test),
                "permutation_importance": float(sum(perm_by_column.get(c, 0.0) for c in cols)),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["delta_r2_when_removed"] = table["r2_full"] - table["r2_without_group"]
        table = table.sort_values("delta_r2_when_removed", ascending=False).reset_index(drop=True)
    summary = {
        "profiled_transactions": len(items),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "target": "illicit logit of the trained detector",
        "r2_full_model": round(full, 4),
        "excluded_from_pool": ["prediction", "counterfactual", "limitations"],
        "salience_variant": "abs_standardized proxy (model-agnostic), to keep the ablation non-circular",
        "note": (
            "The counterfactual group is excluded because it is computed from the detector's own logit "
            "under intervention; its value is established by the faithfulness measurements instead. "
            "Feature salience is scored with the magnitude proxy for the same reason."
        ),
    }
    return table, summary


def citation_utilisation(output_dir: str | Path = "outputs") -> dict[str, Any]:
    """How often the report generator actually cites each evidence group."""
    output_dir = Path(output_dir)
    paths = sorted((output_dir / "metrics").glob("*_btrep_v2_grounded_llm*_report_validation.json"))
    if not paths:
        return {}
    records = read_json(paths[-1])
    total = len(records)
    counts: dict[str, int] = {}
    for record in records:
        for evidence_id in record.get("covered_ids", []):
            counts[evidence_id] = counts.get(evidence_id, 0) + 1
    return {
        "source": paths[-1].name,
        "reports": total,
        "citation_rate": {k: round(v / max(1, total), 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
    }


def build_evidence_value_artifacts(
    evidence_path: str | Path,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    items = read_json(evidence_path)
    table, summary = evidence_value(items)
    if table.empty:
        raise RuntimeError("No observational evidence groups found in the profile.")
    write_table(table, "table15_btrep_component_ablation", output_dir / "tables")
    payload = {
        **summary,
        "evidence_file": Path(evidence_path).name,
        "groups": {
            row["group"]: {
                "fields": int(row["fields"]),
                "delta_r2_when_removed": round(float(row["delta_r2_when_removed"]), 4),
                "r2_group_alone": round(float(row["r2_group_alone"]), 4),
                "permutation_importance": round(float(row["permutation_importance"]), 4),
            }
            for _, row in table.iterrows()
        },
        "generator_citation_utilisation": citation_utilisation(output_dir),
    }
    write_json(payload, output_dir / "metrics" / "btrep_component_ablation_summary.json")
    return payload


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quantify the contribution of each BTREP evidence component.")
    parser.add_argument("--evidence-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    print(json.dumps(build_evidence_value_artifacts(args.evidence_path, args.output_dir), indent=2))
