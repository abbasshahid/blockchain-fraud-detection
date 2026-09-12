"""Faithfulness, fidelity, sufficiency and comprehensiveness for the GNN evidence.

Reviewer 1 pointed out that a report which correctly cites the evidence it was
given has demonstrated *citation validity*, not that the evidence faithfully
describes the detector's decision.  Reviewer 2 asked for the standard
explainability metrics.  Both are addressed here by perturbing the trained
model and measuring what actually changes.

All quantities are computed on the **illicit logit** of the target node rather
than on the probability.  The BTREP sample consists of the highest-risk test
transactions, whose probabilities sit at the saturated end of the sigmoid
(p > 0.999); probability differences there are numerically tiny and would make
a strong explanation look weak.  The logit is the model's actual decision
variable and is not saturated.  Probability-space values are still reported for
readers who prefer them.

Metrics, with ``z`` the illicit logit of the target node:

``comprehensiveness``
    ``z_full - z(top-k features neutralised)``.  Large values mean the cited
    features really carry the decision.
``comprehensiveness_random``
    The same intervention on a random set of k features.  This is the control:
    the gap between the two columns is what shows the ranking is informative
    rather than the masking itself being disruptive.
``sufficiency``
    ``z_full - z(only top-k features retained)``.  Values near zero mean the
    cited features alone reproduce the decision.
``deletion_auc``
    Normalised area under the curve of ``z`` as features are removed in
    attribution order, scaled to the full-model logit; lower is better.
``edge_fidelity``
    ``z_full - z(all incident payment links removed)``, i.e. how much of the
    decision comes from graph context rather than the node's own features.
``counterfactual_flip_k``
    Smallest k on the deletion ladder at which the prediction falls below the
    0.5 decision threshold.  This is the quantity reported to investigators as
    counterfactual evidence.

Two attribution rankings are compared under identical metrics:

``abs_standardized``
    The magnitude of the standardised feature value.  This is the salience
    *proxy* used by BTREP v1 -- model-agnostic, and it never consults the
    detector.
``grad_x_input``
    Gradient of the illicit logit with respect to the node's features, times
    the feature value: a genuine model attribution.

The comparison between the two is the evidence for upgrading BTREP to the
gradient ranking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blockchain_fraud.explain.gnn_probe import GnnProbe
from blockchain_fraud.utils.io import write_json, write_table

DELETION_LADDER = [1, 2, 3, 5, 10, 20, 50, 100]
TOP_K = 5
FLIP_THRESHOLD = 0.5
FLIP_LOGIT = 0.0  # logit(0.5)
METHODS = ["abs_standardized", "grad_x_input"]
RANDOM_SEED = 11


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def attribution_rankings(probe: GnnProbe, nodes: list[int]) -> dict[str, np.ndarray]:
    """Feature indices ordered by descending importance, one row per node."""
    x = probe.x.numpy()
    proxy = np.argsort(-np.abs(x[nodes]), axis=1)
    grads = probe.gradient_attribution(nodes)
    gradient = np.argsort(-np.abs(grads), axis=1)
    rng = np.random.default_rng(RANDOM_SEED)
    random_order = np.stack([rng.permutation(x.shape[1]) for _ in nodes])
    return {
        "abs_standardized": proxy,
        "grad_x_input": gradient,
        "_random": random_order,
        "_grad_values": grads,
    }


def compute_faithfulness(
    model_name: str,
    checkpoint_path: str | Path,
    processed_path: str | Path,
    nodes: list[int],
    top_k: int = TOP_K,
) -> dict[str, object]:
    probe = GnnProbe(model_name, checkpoint_path, processed_path)
    base_logit = probe.base_logits()
    base_prob = probe.base_probabilities()
    groups = probe.independent_groups(nodes)
    rankings = attribution_rankings(probe, nodes)

    edge_removed = probe.batch_edge_removed_probabilities(nodes, groups, return_logits=True)
    random_masks = {node: rankings["_random"][i, :top_k].tolist() for i, node in enumerate(nodes)}
    random_removed = probe.batch_masked_probabilities(random_masks, groups, return_logits=True)

    per_node: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for method in METHODS:
        order = rankings[method]
        deletion: dict[int, dict[int, float]] = {node: {0: float(base_logit[node])} for node in nodes}
        for k in DELETION_LADDER:
            masks = {node: order[i, :k].tolist() for i, node in enumerate(nodes)}
            values = probe.batch_masked_probabilities(masks, groups, return_logits=True)
            for node in nodes:
                deletion[node][k] = values[node]

        keep_masks = {node: order[i, :top_k].tolist() for i, node in enumerate(nodes)}
        only_top = probe.batch_masked_probabilities(keep_masks, groups, keep_only=True, return_logits=True)

        ladder = [0, *DELETION_LADDER]
        for i, node in enumerate(nodes):
            z0 = float(base_logit[node])
            series = [deletion[node][k] for k in ladder]
            auc = float(np.trapezoid(series, ladder) / (ladder[-1] * z0)) if abs(z0) > 1e-9 else float("nan")
            flip_k = next((k for k in DELETION_LADDER if deletion[node][k] < FLIP_LOGIT), None)
            per_node.append(
                {
                    "node_index": node,
                    "method": method,
                    "logit_full": z0,
                    "p_full": float(base_prob[node]),
                    "logit_top_k_removed": deletion[node][top_k],
                    "logit_only_top_k": only_top[node],
                    "logit_random_k_removed": random_removed[node],
                    "logit_edges_removed": edge_removed[node],
                    "p_top_k_removed": _sigmoid(deletion[node][top_k]),
                    "p_edges_removed": _sigmoid(edge_removed[node]),
                    "comprehensiveness": z0 - deletion[node][top_k],
                    "comprehensiveness_random": z0 - random_removed[node],
                    "sufficiency": z0 - only_top[node],
                    "deletion_auc": auc,
                    "edge_fidelity": z0 - edge_removed[node],
                    "counterfactual_flip_k": flip_k if flip_k is not None else np.nan,
                    "counterfactual_flipped": flip_k is not None,
                    "edge_removal_flips": edge_removed[node] < FLIP_LOGIT,
                }
            )
            for k in ladder:
                curves.append(
                    {
                        "node_index": node,
                        "method": method,
                        "k": k,
                        "logit": deletion[node][k],
                        "probability": _sigmoid(deletion[node][k]),
                    }
                )

    return {
        "per_node": pd.DataFrame(per_node),
        "curves": pd.DataFrame(curves),
        "grad_values": rankings["_grad_values"],
        "rankings": {m: rankings[m] for m in METHODS},
        "base_probabilities": base_prob,
        "base_logits": base_logit,
        "feature_names": probe.feature_names,
        "probe": probe,
    }


def summarise(per_node: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "logit_full",
        "p_full",
        "comprehensiveness",
        "comprehensiveness_random",
        "sufficiency",
        "deletion_auc",
        "edge_fidelity",
        "counterfactual_flipped",
        "edge_removal_flips",
    ]
    agg = per_node.groupby("method")[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def build_faithfulness_artifacts(
    output_dir: str | Path = "outputs",
    model_name: str = "graphsage",
    seed: int = 11,
    processed_path: str | Path = "data/processed/elliptic_temporal_v1_all_original_features.pt",
    limit: int = 30,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    run_id = f"elliptic_temporal_v1_{model_name}_all_original_features_seed{seed}"
    predictions = pd.read_csv(output_dir / "predictions" / f"{run_id}_predictions.csv")
    test = predictions[predictions["split"] == "test"]
    nodes = [int(n) for n in test.sort_values("prob_illicit", ascending=False).index[:limit]]

    result = compute_faithfulness(
        model_name,
        output_dir / "checkpoints" / f"{run_id}.pt",
        processed_path,
        nodes,
    )
    per_node: pd.DataFrame = result["per_node"]  # type: ignore[assignment]
    write_table(per_node, f"{run_id}_faithfulness_per_node", output_dir / "tables")
    write_table(result["curves"], f"{run_id}_faithfulness_curves", output_dir / "tables")  # type: ignore[arg-type]
    summary = summarise(per_node)
    write_table(summary, "table11_explanation_faithfulness", output_dir / "tables")

    payload = {
        "model": model_name,
        "seed": seed,
        "nodes_evaluated": len(nodes),
        "top_k": TOP_K,
        "deletion_ladder": DELETION_LADDER,
        "flip_threshold": FLIP_THRESHOLD,
        "measurement_space": "illicit logit (decision variable); probabilities are saturated on this sample",
        "methods": {
            row["method"]: {
                "mean_logit": round(float(row["logit_full_mean"]), 3),
                "mean_probability": round(float(row["p_full_mean"]), 4),
                "comprehensiveness": round(float(row["comprehensiveness_mean"]), 3),
                "comprehensiveness_random_control": round(float(row["comprehensiveness_random_mean"]), 3),
                "sufficiency": round(float(row["sufficiency_mean"]), 3),
                "deletion_auc": round(float(row["deletion_auc_mean"]), 3),
                "counterfactual_flip_rate": round(float(row["counterfactual_flipped_mean"]), 3),
            }
            for _, row in summary.iterrows()
        },
        "edge_fidelity_mean": round(float(per_node["edge_fidelity"].mean()), 3),
        "edge_removal_flip_rate": round(float(per_node["edge_removal_flips"].mean()), 3),
        "interpretation": (
            "Comprehensiveness is the logit drop when the cited features are neutralised, and is only "
            "meaningful against the random-subset control; sufficiency is the residual when only the "
            "cited features are kept; edge fidelity is the drop when the transaction is isolated from "
            "its payment links. Citation validity is measured separately and implies none of these."
        ),
    }
    write_json(payload, output_dir / "metrics" / "faithfulness_summary.json")
    result["summary"] = summary
    result["payload"] = payload
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(build_faithfulness_artifacts()["payload"], indent=2))
