from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blockchain_fraud.utils.hashes import sha256_file
from blockchain_fraud.utils.io import write_json

FEATURES = "elliptic_txs_features.csv"
EDGES = "elliptic_txs_edgelist.csv"
CLASSES = "elliptic_txs_classes.csv"


def locate_raw_files(raw_dir: str | Path) -> dict[str, Path]:
    raw = Path(raw_dir)
    files = {
        "features": raw / FEATURES,
        "edges": raw / EDGES,
        "classes": raw / CLASSES,
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Elliptic files: {missing}")
    return files


def map_class_value(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "illicit"}:
        return 1
    if text in {"2", "licit"}:
        return 0
    if text in {"unknown", "-1", "nan", ""}:
        return -1
    raise ValueError(f"Unsupported class value: {value!r}")


def validate_raw(raw_dir: str | Path, output_dir: str | Path = "outputs") -> dict[str, Any]:
    files = locate_raw_files(raw_dir)
    features = pd.read_csv(files["features"], header=None)
    classes = pd.read_csv(files["classes"])
    edges = pd.read_csv(files["edges"])

    if features.shape[1] < 4:
        raise ValueError("Features file must contain txId, time_step, and feature columns.")
    if not {"txId", "class"}.issubset(classes.columns):
        raise ValueError("Classes file must contain txId and class columns.")
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Edgelist file must contain txId1 and txId2 columns.")

    tx_ids = features.iloc[:, 0].astype(str)
    duplicate_tx_ids = int(tx_ids.duplicated().sum())
    class_ids = classes["txId"].astype(str)
    edge_src = edges["txId1"].astype(str)
    edge_dst = edges["txId2"].astype(str)
    feature_id_set = set(tx_ids)

    missing_class_ids = int((~class_ids.isin(feature_id_set)).sum())
    missing_edge_endpoints = int((~edge_src.isin(feature_id_set)).sum() + (~edge_dst.isin(feature_id_set)).sum())
    duplicate_edges = int(edges.duplicated().sum())
    self_loops = int((edge_src == edge_dst).sum())

    y = classes["class"].map(map_class_value)
    feature_values = features.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    missing_values = int(feature_values.isna().sum().sum())
    finite_mask = np.isfinite(feature_values.to_numpy(dtype=np.float64, copy=False))
    non_finite_values = int((~finite_mask).sum())

    time_steps = pd.to_numeric(features.iloc[:, 1], errors="raise").astype(int)
    row_count = int(features.shape[0])
    report: dict[str, Any] = {
        "raw_dir": str(Path(raw_dir).resolve()),
        "files": {k: str(v.resolve()) for k, v in files.items()},
        "file_hashes": {k: sha256_file(v) for k, v in files.items()},
        "counts": {
            "nodes": row_count,
            "edges": int(edges.shape[0]),
            "features_per_node": int(features.shape[1] - 2),
            "classes_rows": int(classes.shape[0]),
            "unique_time_steps": int(time_steps.nunique()),
            "min_time_step": int(time_steps.min()),
            "max_time_step": int(time_steps.max()),
        },
        "labels": {
            "illicit": int((y == 1).sum()),
            "licit": int((y == 0).sum()),
            "unknown": int((y == -1).sum()),
        },
        "integrity": {
            "duplicate_tx_ids": duplicate_tx_ids,
            "missing_class_ids": missing_class_ids,
            "missing_edge_endpoints": missing_edge_endpoints,
            "duplicate_edges": duplicate_edges,
            "self_loops": self_loops,
            "missing_feature_values": missing_values,
            "non_finite_feature_values": non_finite_values,
            "features_header_present": False,
            "classes_header_present": True,
            "edges_header_present": True,
        },
        "feature_groups": {
            "time_step_column_index": 1,
            "feature_column_start_index": 2,
            "local_feature_count_default": 93,
            "aggregated_feature_count_default": max(0, int(features.shape[1] - 2 - 93)),
        },
    }

    hard_failures = {
        key: value
        for key, value in report["integrity"].items()
        if key in {"duplicate_tx_ids", "missing_class_ids", "missing_edge_endpoints", "missing_feature_values", "non_finite_feature_values"}
        and value != 0
    }
    output_path = Path(output_dir) / "metrics" / "data_validation.json"
    write_json(report, output_path)
    if hard_failures:
        raise ValueError(f"Raw data integrity checks failed: {hard_failures}. Report written to {output_path}")
    if not math.isclose(sum(report["labels"].values()), classes.shape[0]):
        raise ValueError("Class mapping count mismatch.")
    return report

