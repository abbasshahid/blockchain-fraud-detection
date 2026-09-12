"""Blockchain Transaction Risk Evidence Profile (BTREP).

A BTREP record is the complete, machine-checkable evidence package for one
transaction.  Every field is derived from persisted model outputs, the observed
transaction graph, or a counterfactual re-execution of the trained detector --
never from hidden validation or test labels.

Version history
---------------
``btrep_v1``
    Seven evidence groups: prediction, structural, temporal, risk exposure,
    motifs, feature salience, limitations.  Feature salience was a *proxy*: the
    largest standardised feature values, which never consult the detector.
``btrep_v2``
    Adds an eighth group, ``counterfactual`` (``cf_1``), computed by
    re-running the trained model under interventions -- removing the
    transaction's payment links, and neutralising its most attributed features
    -- and upgrades feature salience from the magnitude proxy to
    gradient x input attribution of the illicit logit.  Both changes exist so
    that the generated report can state what would have happened had the
    evidence been absent, which is what an investigator actually needs, and so
    that the salience field survives the faithfulness tests in
    :mod:`blockchain_fraud.explain.faithfulness`.

``btrep_v2`` degrades gracefully: when no checkpoint is supplied the profile is
built with the v1 proxy salience and without the counterfactual group.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blockchain_fraud.utils.io import ensure_dir, load_graph, write_json

EVIDENCE_VERSION = "btrep_v2"
TOP_FEATURES = 5
FLIP_LADDER = [1, 2, 3, 5, 10, 20, 50, 100]


def _weak_component_sizes(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, d in edge_index.T:
        adj[int(s)].append(int(d))
        adj[int(d)].append(int(s))
    comp = np.zeros(num_nodes, dtype=np.int64)
    seen = np.zeros(num_nodes, dtype=bool)
    for i in range(num_nodes):
        if seen[i]:
            continue
        q: deque[int] = deque([i])
        seen[i] = True
        nodes: list[int] = []
        while q:
            u = q.popleft()
            nodes.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comp[nodes] = len(nodes)
    return comp


def _counterfactual_evidence(
    model_name: str,
    checkpoint_path: str | Path,
    processed_path: str | Path,
    nodes: list[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    """Re-run the detector under interventions; returns evidence and salience order."""
    from blockchain_fraud.explain.gnn_probe import GnnProbe

    probe = GnnProbe(model_name, checkpoint_path, processed_path)
    base_logit = probe.base_logits()
    groups = probe.independent_groups(nodes)
    grads = probe.gradient_attribution(nodes)
    order = {node: np.argsort(-np.abs(grads[i])).tolist() for i, node in enumerate(nodes)}

    isolated = probe.batch_edge_removed_probabilities(nodes, groups, return_logits=True)
    top1 = probe.batch_masked_probabilities({n: order[n][:1] for n in nodes}, groups, return_logits=True)
    ladder: dict[int, dict[int, float]] = {n: {} for n in nodes}
    for k in FLIP_LADDER:
        values = probe.batch_masked_probabilities({n: order[n][:k] for n in nodes}, groups, return_logits=True)
        for node in nodes:
            ladder[node][k] = values[node]

    def _p(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    evidence: dict[int, dict[str, Any]] = {}
    for node in nodes:
        z0 = float(base_logit[node])
        z_iso = isolated[node]
        flip_k = next((k for k in FLIP_LADDER if ladder[node][k] < 0.0), None)
        evidence[node] = {
            "id": "cf_1",
            "method": "model_re_execution",
            "probability_without_payment_links": round(_p(z_iso), 4),
            "logit_drop_without_payment_links": round(z0 - z_iso, 3),
            "share_of_decision_from_graph_context": round(float(np.clip((z0 - z_iso) / z0, 0.0, 1.0)), 3)
            if abs(z0) > 1e-9
            else None,
            "isolating_transaction_flips_prediction": bool(z_iso < 0.0),
            "probability_without_top_attributed_feature": round(_p(top1[node]), 4),
            "minimum_attributed_features_to_flip": flip_k,
            "flip_search_ladder": FLIP_LADDER,
        }
    return evidence, order


def build_btrep(
    processed_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path = "outputs",
    limit: int = 50,
    high_risk_threshold: float = 0.7,
    model_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    output_suffix: str = "_btrep",
    selection: str = "top",
) -> list[dict[str, Any]]:
    """Build evidence profiles for test transactions.

    ``selection`` controls which transactions are profiled. ``"top"`` takes the
    highest-risk transactions, which is the investigation-facing use case.
    ``"stratified"`` samples evenly across risk deciles, which is what the
    evidence-value analysis needs: on the top-risk sample the detector's logit
    is nearly constant, so no evidence group can be shown to track it.
    """
    graph = load_graph(processed_path)
    pred = pd.read_csv(prediction_path)
    y = graph["y"].numpy()
    time_step = graph["time_step"].numpy()
    edge_index = graph["edge_index"].numpy()
    num_nodes = len(y)
    probs = pred["prob_illicit"].to_numpy(dtype=float)

    in_neighbors: dict[int, list[int]] = defaultdict(list)
    out_neighbors: dict[int, list[int]] = defaultdict(list)
    for src, dst in edge_index.T:
        out_neighbors[int(src)].append(int(dst))
        in_neighbors[int(dst)].append(int(src))

    indeg = np.asarray([len(in_neighbors[i]) for i in range(num_nodes)])
    outdeg = np.asarray([len(out_neighbors[i]) for i in range(num_nodes)])
    total_degree = indeg + outdeg
    train_nodes = graph["train_mask"].numpy().astype(bool)
    train_indeg_p75 = float(np.percentile(indeg[train_nodes], 75))
    train_outdeg_p75 = float(np.percentile(outdeg[train_nodes], 75))
    train_degree_p25 = float(np.percentile(total_degree[train_nodes], 25))
    train_degree_p90 = float(np.percentile(total_degree[train_nodes], 90))
    component_sizes = _weak_component_sizes(num_nodes, edge_index)

    test_labeled = np.where(graph["test_mask"].numpy().astype(bool))[0]
    order = test_labeled[np.argsort(-probs[test_labeled])]
    if selection == "stratified":
        rng = np.random.default_rng(11)
        deciles = np.array_split(order, 10)
        per_bin = max(1, limit // len(deciles))
        picked: list[int] = []
        for chunk in deciles:
            take = min(per_bin, len(chunk))
            picked.extend(int(v) for v in rng.choice(chunk, size=take, replace=False))
        selected = sorted(set(picked))[:limit]
    elif selection == "top":
        selected = [int(n) for n in order[:limit]]
    else:
        raise ValueError(f"Unknown selection strategy: {selection}")
    max_time = max(1, int(time_step.max()))

    counterfactuals: dict[int, dict[str, Any]] = {}
    salience_order: dict[int, list[int]] = {}
    salience_method = "abs_standardized_proxy"
    if model_name and checkpoint_path and Path(checkpoint_path).exists():
        counterfactuals, salience_order = _counterfactual_evidence(
            model_name, checkpoint_path, processed_path, selected
        )
        salience_method = "gradient_x_input"

    evidence: list[dict[str, Any]] = []
    for node in selected:
        preds = set(in_neighbors[int(node)])
        succs = set(out_neighbors[int(node)])
        one_hop = preds | succs
        two_hop = set(one_hop)
        for n in list(one_hop)[:250]:
            two_hop.update(in_neighbors[n])
            two_hop.update(out_neighbors[n])
        neighbor_probs = probs[list(one_hop)] if one_hop else np.asarray([], dtype=float)
        train_history_neighbors = [n for n in one_hop if train_nodes[n] and time_step[n] < time_step[node]]
        train_history_illicit = int(sum(y[n] == 1 for n in train_history_neighbors))
        motifs: list[str] = []
        if indeg[node] >= train_indeg_p75 and outdeg[node] <= train_degree_p25:
            motifs.append("fan_in_like")
        if outdeg[node] >= train_outdeg_p75 and indeg[node] <= train_degree_p25:
            motifs.append("fan_out_like")
        if indeg[node] <= 1 and outdeg[node] <= 1:
            motifs.append("chain_like")
        if total_degree[node] <= train_degree_p25:
            motifs.append("isolated_or_local")
        if total_degree[node] >= train_degree_p90:
            motifs.append("dense_neighborhood_like")

        if node in salience_order:
            top_features_idx = salience_order[node][:TOP_FEATURES]
        else:
            top_features_idx = np.argsort(-np.abs(graph["x"][node].numpy()))[:TOP_FEATURES].tolist()

        record: dict[str, Any] = {
            "transaction_id": graph["tx_id"][int(node)],
            "node_index": int(node),
            "time_step": int(time_step[node]),
            "prediction": "illicit" if probs[node] >= 0.5 else "licit",
            "calibrated_probability": float(probs[node]),
            "evidence_version": EVIDENCE_VERSION if counterfactuals else "btrep_v1",
            "evidence": {
                "prediction": {"id": "pred_1", "probability": float(probs[node]), "label": "illicit" if probs[node] >= 0.5 else "licit"},
                "structural": {
                    "id": "struct_1",
                    "in_degree": int(indeg[node]),
                    "out_degree": int(outdeg[node]),
                    "total_degree": int(total_degree[node]),
                    "one_hop_size": int(len(one_hop)),
                    "two_hop_size_capped": int(min(len(two_hop), 1000)),
                    "component_size": int(component_sizes[node]),
                },
                "temporal": {
                    "id": "temp_1",
                    "relative_time": float(time_step[node] / max_time),
                    "time_step": int(time_step[node]),
                    "degree_train_percentile_proxy": float((total_degree[train_nodes] <= total_degree[node]).mean()),
                },
                "risk_exposure": {
                    "id": "risk_1",
                    "mean_neighbor_risk": float(neighbor_probs.mean()) if neighbor_probs.size else 0.0,
                    "max_neighbor_risk": float(neighbor_probs.max()) if neighbor_probs.size else 0.0,
                    "high_risk_neighbor_count": int((neighbor_probs >= high_risk_threshold).sum()) if neighbor_probs.size else 0,
                    "training_history_labeled_neighbors": int(len(train_history_neighbors)),
                    "training_history_illicit_neighbors": train_history_illicit,
                },
                "motifs": {"id": "motif_1", "values": motifs},
                "feature_salience": {
                    "id": "feat_1",
                    "method": salience_method,
                    "top_features": [
                        {"name": graph["feature_names"][int(i)], "standardized_value": float(graph["x"][node, int(i)].item())}
                        for i in top_features_idx
                    ],
                },
                "limitations": {
                    "id": "limit_1",
                    "text": "Feature semantics are anonymized; this is model evidence, not verified criminal attribution.",
                },
            },
        }
        if node in counterfactuals:
            record["evidence"]["counterfactual"] = counterfactuals[node]
        evidence.append(record)

    stem = Path(prediction_path).stem.replace("_predictions", "")
    out_path = Path(output_dir) / "explanations" / f"{stem}{output_suffix}.json"
    ensure_dir(out_path.parent)
    write_json(evidence, out_path)
    write_json(
        {
            "evidence_version": EVIDENCE_VERSION if counterfactuals else "btrep_v1",
            "salience_method": salience_method,
            "high_risk_threshold": high_risk_threshold,
            "train_indeg_p75": train_indeg_p75,
            "train_outdeg_p75": train_outdeg_p75,
            "train_degree_p25": train_degree_p25,
            "train_degree_p90": train_degree_p90,
            "top_features": TOP_FEATURES,
            "flip_ladder": FLIP_LADDER,
        },
        Path(output_dir) / "metrics" / "btrep_thresholds.json",
    )
    return evidence
