from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from blockchain_fraud.data.feature_dictionary import feature_groups, feature_names
from blockchain_fraud.data.temporal_split import make_temporal_split
from blockchain_fraud.data.validate_raw import locate_raw_files, map_class_value
from blockchain_fraud.utils.hashes import sha256_file
from blockchain_fraud.utils.io import ensure_dir, save_graph, write_json


def _mask_from_steps(time_steps: np.ndarray, steps: list[int]) -> np.ndarray:
    return np.isin(time_steps, np.asarray(steps, dtype=time_steps.dtype))


def preprocess(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(config.get("raw_dir", "elliptic_bitcoin_dataset"))
    processed_dir = ensure_dir(config.get("processed_dir", "data/processed"))
    output_dir = Path(config.get("output_dir", "outputs"))
    feature_mode = config.get("feature_mode", "all_original_features")
    experiment_name = config.get("experiment_name", "elliptic_temporal_v1")
    local_feature_count = int(config.get("local_feature_count", 93))
    split_cfg = config.get("split", {})

    files = locate_raw_files(raw_dir)
    features = pd.read_csv(files["features"], header=None)
    classes = pd.read_csv(files["classes"])
    edges = pd.read_csv(files["edges"])

    tx_id = features.iloc[:, 0].astype(str).to_numpy()
    time_step = pd.to_numeric(features.iloc[:, 1], errors="raise").astype(int).to_numpy()
    all_x = features.iloc[:, 2:].to_numpy(dtype=np.float32)
    groups = feature_groups(all_x.shape[1], local_feature_count)
    if feature_mode == "local_only":
        selected_idx = groups["local"]
    elif feature_mode == "all_original_features":
        selected_idx = list(range(all_x.shape[1]))
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    x = all_x[:, selected_idx]

    y_by_tx = {str(row.txId): map_class_value(row[1]) for row in classes.itertuples(index=False)}
    y = np.asarray([y_by_tx[str(t)] for t in tx_id], dtype=np.int64)

    split = make_temporal_split(
        time_step,
        train_fraction=float(split_cfg.get("train_fraction", 0.60)),
        val_fraction=float(split_cfg.get("val_fraction", 0.20)),
    )
    train_period_mask = _mask_from_steps(time_step, split["train_steps"])
    train_mask = train_period_mask & (y >= 0)
    val_mask = _mask_from_steps(time_step, split["val_steps"]) & (y >= 0)
    test_mask = _mask_from_steps(time_step, split["test_steps"]) & (y >= 0)

    scaler = StandardScaler()
    scaler.fit(x[train_period_mask])
    x_scaled = scaler.transform(x).astype(np.float32)

    tx_to_idx = {tid: i for i, tid in enumerate(tx_id)}
    src = edges["txId1"].astype(str).map(tx_to_idx).to_numpy()
    dst = edges["txId2"].astype(str).map(tx_to_idx).to_numpy()
    if pd.isna(src).any() or pd.isna(dst).any():
        raise ValueError("Edge endpoint missing from feature transaction IDs.")
    edge_index = np.vstack([src.astype(np.int64), dst.astype(np.int64)])

    graph = {
        "x": torch.from_numpy(x_scaled),
        "edge_index": torch.from_numpy(edge_index),
        "y": torch.from_numpy(y),
        "time_step": torch.from_numpy(time_step.astype(np.int64)),
        "tx_id": tx_id.tolist(),
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
        "test_mask": torch.from_numpy(test_mask),
        "feature_names": [feature_names(all_x.shape[1])[i] for i in selected_idx],
        "feature_mode": feature_mode,
        "selected_feature_indices": selected_idx,
        "feature_groups": groups,
        "scaler": {
            "method": "standard",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "fit_node_count": int(train_period_mask.sum()),
            "fit_time_steps": split["train_steps"],
        },
        "metadata": {
            "dataset": "elliptic",
            "split_version": config.get("split_version", "temporal_60_20_20_v1"),
            "raw_hashes": {k: sha256_file(v) for k, v in files.items()},
            "canonical_directed_edges": True,
        },
    }

    stem = f"{experiment_name}_{feature_mode}"
    graph_path = processed_dir / f"{stem}.pt"
    save_graph(graph, graph_path)

    manifest = {
        "processed_graph": str(graph_path),
        "feature_mode": feature_mode,
        "split_version": config.get("split_version", "temporal_60_20_20_v1"),
        **split,
        "counts": split_counts(graph),
        "scaler": graph["scaler"],
    }
    write_json(manifest, output_dir / "metrics" / f"{stem}_split_manifest.json")
    return manifest


def split_counts(graph: dict[str, Any]) -> dict[str, Any]:
    y = graph["y"].numpy()
    time_step = graph["time_step"].numpy()
    edge_index = graph["edge_index"].numpy()
    out: dict[str, Any] = {}
    for split_name, mask_name in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        labeled_mask = graph[mask_name].numpy().astype(bool)
        period_steps = np.unique(time_step[labeled_mask]).tolist()
        node_period_mask = np.isin(time_step, period_steps)
        visible_edges = node_period_mask[edge_index[0]] & node_period_mask[edge_index[1]]
        out[split_name] = {
            "time_steps": [int(s) for s in period_steps],
            "total_nodes_in_period": int(node_period_mask.sum()),
            "labeled_nodes": int(labeled_mask.sum()),
            "illicit": int(((y == 1) & labeled_mask).sum()),
            "licit": int(((y == 0) & labeled_mask).sum()),
            "unknown": int(((y == -1) & node_period_mask).sum()),
            "edges_visible_within_period": int(visible_edges.sum()),
            "illicit_ratio": float(((y == 1) & labeled_mask).sum() / max(1, labeled_mask.sum())),
        }
    return out

