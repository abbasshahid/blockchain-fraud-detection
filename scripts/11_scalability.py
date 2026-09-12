"""Computational cost and scalability measurements.

Reviewer 2 asked for training time, inference cost, memory and scalability
information beyond the runtime column already in the efficiency table.  We
measure two things:

1. **Per-model cost at full scale** -- wall-clock training and inference time,
   peak resident memory attributable to the run, parameter count and
   per-transaction inference throughput.
2. **Scaling behaviour** -- the same measurements on node-induced subgraphs at
   25%, 50%, 75% and 100% of the transaction graph, which shows how cost grows
   with graph size.  Subgraphs are sampled by *time step* rather than at random
   so that each point remains a coherent, leakage-aware temporal graph rather
   than a shredded one.

Memory is measured with ``tracemalloc`` for Python-level allocations plus the
process peak working set, because the sparse adjacency and the dense feature
matrix dominate and are allocated outside Python's allocator.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from blockchain_fraud.config import load_yaml  # noqa: E402
from blockchain_fraud.models.gnn import GCN, GraphSAGE, SimpleGAT  # noqa: E402
from blockchain_fraud.models.graph_utils import normalized_sparse_adj, prepare_edges  # noqa: E402
from blockchain_fraud.utils.io import load_graph, write_json, write_table  # noqa: E402

import pandas as pd  # noqa: E402

PROCESSED = "data/processed/elliptic_temporal_v1_all_original_features.pt"
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
GNNS = ["gcn", "graphsage", "gat"]


def _peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024**2  # type: ignore[attr-defined]
    except Exception:
        return float("nan")


def _build(name: str, in_channels: int, hidden: int, dropout: float) -> torch.nn.Module:
    if name == "gcn":
        return GCN(in_channels, hidden, dropout)
    if name == "graphsage":
        return GraphSAGE(in_channels, hidden, dropout)
    return SimpleGAT(in_channels, hidden, dropout)


def _forward(name: str, model: torch.nn.Module, x, adj, edge_index):
    if name in {"gcn", "graphsage"}:
        return model(x, adj)
    return model(x, edge_index)


def subgraph_by_time(graph: dict, fraction: float) -> dict:
    """Node-induced subgraph over the earliest ``fraction`` of time steps."""
    time_step = graph["time_step"].numpy()
    steps = np.unique(time_step)
    keep_steps = steps[: max(3, int(round(fraction * len(steps))))]
    keep = np.isin(time_step, keep_steps)
    index_map = -np.ones(len(keep), dtype=np.int64)
    index_map[keep] = np.arange(keep.sum())
    edges = graph["edge_index"].numpy()
    edge_keep = keep[edges[0]] & keep[edges[1]]
    return {
        "x": graph["x"][torch.from_numpy(keep)],
        "edge_index": torch.from_numpy(np.vstack([index_map[edges[0][edge_keep]], index_map[edges[1][edge_keep]]])),
        "y": graph["y"][torch.from_numpy(keep)],
        "train_mask": graph["train_mask"][torch.from_numpy(keep)],
    }


def measure(model_name: str, graph: dict, epochs: int, seed: int = 11) -> dict[str, float]:
    torch.manual_seed(seed)
    cfg = load_yaml(f"configs/{model_name}.yaml").get("training", {})
    x = graph["x"]
    y = graph["y"].float()
    num_nodes = x.size(0)
    train_mask = graph["train_mask"].bool()

    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()

    edge_index = prepare_edges(graph["edge_index"], num_nodes, bidirectional=True, self_loops=True)
    adj = None
    if model_name == "gcn":
        adj = normalized_sparse_adj(edge_index, num_nodes, "gcn")
    elif model_name == "graphsage":
        adj = normalized_sparse_adj(edge_index, num_nodes, "mean")

    pos = int((y[train_mask] == 1).sum().item())
    neg = int((y[train_mask] == 0).sum().item())
    model = _build(model_name, x.size(1), int(cfg.get("hidden_channels", 32)), float(cfg.get("dropout", 0.35)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 0.01)))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(1, pos)], dtype=torch.float32))

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model_name, model, x, adj, edge_index)
        loss_fn(logits[train_mask], y[train_mask]).backward()
        optimizer.step()
    train_seconds = time.perf_counter() - started

    model.eval()
    infer_started = time.perf_counter()
    with torch.no_grad():
        _forward(model_name, model, x, adj, edge_index)
    inference_seconds = time.perf_counter() - infer_started

    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Torch allocates outside Python's allocator, so tracemalloc under-reports.
    # The two tensors that dominate residency are measured directly.
    feature_bytes = x.element_size() * x.nelement()
    adjacency_bytes = 0
    if adj is not None:
        adjacency_bytes = adj._values().element_size() * adj._values().nelement()
        adjacency_bytes += adj._indices().element_size() * adj._indices().nelement()
    else:
        adjacency_bytes = edge_index.element_size() * edge_index.nelement()

    return {
        "nodes": int(num_nodes),
        "edges_after_preparation": int(edge_index.size(1)),
        "epochs": epochs,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / max(1, epochs),
        "inference_seconds": inference_seconds,
        "inference_us_per_node": 1e6 * inference_seconds / max(1, num_nodes),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "python_peak_mib": python_peak / 1024**2,
        "feature_matrix_mib": feature_bytes / 1024**2,
        "adjacency_mib": adjacency_bytes / 1024**2,
        "resident_tensor_mib": (feature_bytes + adjacency_bytes) / 1024**2,
        "process_peak_rss_mib": _peak_rss_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure training/inference cost and graph-size scaling.")
    parser.add_argument("--processed-path", default=PROCESSED)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Fixed epoch budget so points are comparable.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    full = load_graph(args.processed_path)

    warmup = subgraph_by_time(full, 0.25)
    for model_name in GNNS:
        measure(model_name, warmup, 2)
    print("warm-up complete; timings below exclude start-up cost", flush=True)

    rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        sub = subgraph_by_time(full, fraction) if fraction < 1.0 else {
            "x": full["x"],
            "edge_index": full["edge_index"],
            "y": full["y"],
            "train_mask": full["train_mask"],
        }
        for model_name in GNNS:
            stats = measure(model_name, sub, args.epochs)
            rows.append({"fraction": fraction, "model": model_name, **stats})
            print(
                f"[{model_name}] fraction={fraction} nodes={stats['nodes']} "
                f"s/epoch={stats['seconds_per_epoch']:.3f} "
                f"infer={stats['inference_seconds']:.3f}s "
                f"tensors={stats['resident_tensor_mib']:.0f}MiB",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_table(df, "table16_scalability", output_dir / "tables")

    full_scale = df[df["fraction"] == 1.0]
    payload = {
        "epoch_budget": args.epochs,
        "fractions": FRACTIONS,
        "hardware_note": "Full-batch CPU training; see outputs/metrics/environment.json for the machine.",
        "full_scale": {
            r.model: {
                "nodes": int(r.nodes),
                "edges_after_preparation": int(r.edges_after_preparation),
                "seconds_per_epoch": round(float(r.seconds_per_epoch), 3),
                "inference_seconds": round(float(r.inference_seconds), 3),
                "inference_us_per_node": round(float(r.inference_us_per_node), 2),
                "parameters": int(r.parameters),
                "resident_tensor_mib": round(float(r.resident_tensor_mib), 1),
            }
            for r in full_scale.itertuples(index=False)
        },
        "scaling_exponent_seconds_per_epoch": {
            model: round(
                float(
                    np.polyfit(
                        np.log(block["nodes"].to_numpy(dtype=float)),
                        np.log(block["seconds_per_epoch"].to_numpy(dtype=float)),
                        1,
                    )[0]
                ),
                3,
            )
            for model, block in df.groupby("model")
        },
    }
    write_json(payload, output_dir / "metrics" / "scalability_summary.json")
    print(json.dumps(payload["scaling_exponent_seconds_per_epoch"], indent=2))


if __name__ == "__main__":
    main()
