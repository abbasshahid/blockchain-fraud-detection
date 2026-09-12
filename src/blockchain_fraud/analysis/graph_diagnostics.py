"""Graph-quality diagnostics: why does a feature-only baseline win?

Both reviewers asked for a deeper explanation of the XGBoost result.  Rather
than speculating, we measure the three properties that determine whether
message passing can add anything on top of the Elliptic feature vector:

1. **Neighbourhood availability** -- how many test transactions have any
   neighbour at all, and how many have a *labelled* neighbour.  A node of
   degree one whose single neighbour is unlabelled receives no usable signal.
2. **Label homophily** -- whether connected labelled transactions share a
   class.  Message passing helps when they do.
3. **Feature redundancy** -- how much of the 72 aggregated Elliptic features can
   be reconstructed by ridge regression from the *mean of the neighbours' local
   features*.  A high R^2 means the original feature vector already contains a
   one-hop aggregation, so an extra learned aggregation is largely duplicated
   work while adding variance.

Finally we split XGBoost's gain-based importance between the local block
(columns 1--93) and the aggregated block (columns 94--165).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from blockchain_fraud.utils.io import load_graph, write_json, write_table

PROCESSED = "data/processed/elliptic_temporal_v1_all_original_features.pt"
LOCAL_FEATURE_COUNT = 93
REFERENCE_SEED = 11


def _mean_neighbour_operator(edge_index: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """Row-normalised undirected adjacency without self-loops."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (dst, src)), shape=(num_nodes, num_nodes)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    nonzero = deg > 0
    inv[nonzero] = 1.0 / deg[nonzero]
    return sp.diags(inv) @ adj


def neighbourhood_availability(graph: dict) -> dict[str, object]:
    y = graph["y"].numpy()
    edge_index = graph["edge_index"].numpy()
    n = len(y)
    labelled = y >= 0
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1] * 2, dtype=np.float32),
         (np.concatenate([edge_index[1], edge_index[0]]), np.concatenate([edge_index[0], edge_index[1]]))),
        shape=(n, n),
    ).tocsr()
    adj.data[:] = 1.0
    degree = np.asarray(adj.sum(axis=1)).ravel()
    labelled_neighbours = np.asarray(adj @ labelled.astype(np.float32)).ravel()
    illicit_neighbours = np.asarray(adj @ (y == 1).astype(np.float32)).ravel()

    out: dict[str, object] = {}
    for split in ["train", "val", "test"]:
        mask = graph[f"{split}_mask"].numpy().astype(bool)
        illicit_mask = mask & (y == 1)
        out[split] = {
            "labelled_nodes": int(mask.sum()),
            "mean_degree": round(float(degree[mask].mean()), 3),
            "share_degree_zero": round(float((degree[mask] == 0).mean()), 4),
            "share_degree_one": round(float((degree[mask] <= 1).mean()), 4),
            "share_with_labelled_neighbour": round(float((labelled_neighbours[mask] > 0).mean()), 4),
            "illicit_share_degree_one": round(float((degree[illicit_mask] <= 1).mean()), 4),
            "illicit_share_with_labelled_neighbour": round(float((labelled_neighbours[illicit_mask] > 0).mean()), 4),
            "illicit_share_with_illicit_neighbour": round(float((illicit_neighbours[illicit_mask] > 0).mean()), 4),
        }
    return out


def label_homophily(graph: dict) -> dict[str, object]:
    y = graph["y"].numpy()
    edge_index = graph["edge_index"].numpy()
    both_labelled = (y[edge_index[0]] >= 0) & (y[edge_index[1]] >= 0)
    src_y = y[edge_index[0]][both_labelled]
    dst_y = y[edge_index[1]][both_labelled]
    total_edges = int(edge_index.shape[1])
    n_ll = int(both_labelled.sum())
    same = int((src_y == dst_y).sum())
    illicit_illicit = int(((src_y == 1) & (dst_y == 1)).sum())
    illicit_any = int(((src_y == 1) | (dst_y == 1)).sum())
    return {
        "total_directed_edges": total_edges,
        "edges_with_both_endpoints_labelled": n_ll,
        "share_of_edges_fully_labelled": round(n_ll / total_edges, 4),
        "edge_homophily_labelled": round(same / max(1, n_ll), 4),
        "illicit_illicit_edges": illicit_illicit,
        "share_illicit_edges_that_are_illicit_illicit": round(illicit_illicit / max(1, illicit_any), 4),
    }


def feature_redundancy(graph: dict, local_feature_count: int = LOCAL_FEATURE_COUNT, ridge_alpha: float = 1.0) -> dict[str, object]:
    """Can the aggregated Elliptic block be reconstructed from neighbours' local features?"""
    x = graph["x"].numpy()
    edge_index = graph["edge_index"].numpy()
    n = x.shape[0]
    local = x[:, :local_feature_count]
    aggregated = x[:, local_feature_count:]
    neighbour_mean_local = np.asarray(_mean_neighbour_operator(edge_index, n) @ local)

    train_mask = graph["train_mask"].numpy().astype(bool)
    test_mask = graph["test_mask"].numpy().astype(bool)

    # Within-period holdout isolates "is the mapping learnable" from the
    # temporal drift that a train -> test evaluation also picks up.
    rng = np.random.default_rng(0)
    train_idx = np.flatnonzero(train_mask)
    rng.shuffle(train_idx)
    cut = int(0.8 * len(train_idx))
    fit_idx, holdout_idx = train_idx[:cut], train_idx[cut:]

    def _fit_eval(feature_matrix: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
        model = Ridge(alpha=ridge_alpha)
        model.fit(feature_matrix[fit_idx], aggregated[fit_idx])
        return r2_score(aggregated[eval_idx], model.predict(feature_matrix[eval_idx]), multioutput="raw_values")

    design = np.hstack([local, neighbour_mean_local])
    within = _fit_eval(design, holdout_idx)
    across = _fit_eval(design, np.flatnonzero(test_mask))
    local_only_within = _fit_eval(local, holdout_idx)

    return {
        "aggregated_feature_count": int(aggregated.shape[1]),
        "within_train_period_holdout": {
            "median_r2_from_own_local_only": round(float(np.median(local_only_within)), 4),
            "median_r2_from_local_plus_neighbour_mean": round(float(np.median(within)), 4),
            "share_targets_r2_above_0.5": round(float((within > 0.5).mean()), 4),
            "share_targets_r2_above_0.9": round(float((within > 0.9).mean()), 4),
            "neighbour_mean_median_gain": round(float(np.median(within) - np.median(local_only_within)), 4),
        },
        "across_period_test": {
            "median_r2_from_local_plus_neighbour_mean": round(float(np.median(across)), 4),
            "share_targets_r2_above_0.5": round(float((across > 0.5).mean()), 4),
        },
        "note": (
            "A high within-period R^2 with a collapsing across-period R^2 indicates that the "
            "aggregated Elliptic block is reconstructable from one-hop local statistics inside a "
            "period but drifts between periods."
        ),
    }


def xgboost_importance_split(
    output_dir: str | Path = "outputs",
    seed: int = REFERENCE_SEED,
    local_feature_count: int = LOCAL_FEATURE_COUNT,
) -> dict[str, object]:
    model_path = Path(output_dir) / "checkpoints" / f"elliptic_temporal_v1_xgboost_all_original_features_seed{seed}.json"
    if not model_path.exists():
        return {}
    booster = xgb.Booster()
    booster.load_model(model_path)
    gain = booster.get_score(importance_type="total_gain")
    names = booster.feature_names or []
    index_of = {name: i for i, name in enumerate(names)}
    local_gain = sum(v for k, v in gain.items() if index_of.get(k, 0) < local_feature_count)
    agg_gain = sum(v for k, v in gain.items() if index_of.get(k, 0) >= local_feature_count)
    total = local_gain + agg_gain
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:10]
    return {
        "seed": seed,
        "local_feature_count": local_feature_count,
        "aggregated_feature_count": len(names) - local_feature_count,
        "local_gain_share": round(local_gain / max(1e-9, total), 4),
        "aggregated_gain_share": round(agg_gain / max(1e-9, total), 4),
        "top_features_by_total_gain": [
            {"feature": k, "block": "local" if index_of.get(k, 0) < local_feature_count else "aggregated", "total_gain": round(v, 1)}
            for k, v in top
        ],
    }


def build_graph_diagnostics(output_dir: str | Path = "outputs", processed_path: str | Path = PROCESSED) -> dict[str, object]:
    output_dir = Path(output_dir)
    graph = load_graph(processed_path)
    payload = {
        "neighbourhood_availability": neighbourhood_availability(graph),
        "label_homophily": label_homophily(graph),
        "feature_redundancy": feature_redundancy(graph),
        "xgboost_importance_split": xgboost_importance_split(output_dir),
    }
    write_json(payload, output_dir / "metrics" / "graph_diagnostics.json")

    rows = [{"split": split, **stats} for split, stats in payload["neighbourhood_availability"].items()]
    write_table(pd.DataFrame(rows), "table10_graph_diagnostics", output_dir / "tables")
    return payload


if __name__ == "__main__":
    import json

    print(json.dumps(build_graph_diagnostics(), indent=2))
