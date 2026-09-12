from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from blockchain_fraud.models.gnn import GCN, GraphSAGE, SimpleGAT
from blockchain_fraud.models.graph_utils import normalized_sparse_adj, prepare_edges
from blockchain_fraud.seed import set_seed
from blockchain_fraud.training.evaluator import binary_metrics
from blockchain_fraud.utils.device import device_report, resolve_device
from blockchain_fraud.utils.io import ensure_dir, load_graph, write_json


def _build_model(name: str, in_channels: int, hidden: int, dropout: float) -> nn.Module:
    if name == "gcn":
        return GCN(in_channels, hidden, dropout)
    if name == "graphsage":
        return GraphSAGE(in_channels, hidden, dropout)
    if name == "gat":
        return SimpleGAT(in_channels, hidden, dropout)
    raise ValueError(f"Unsupported GNN model: {name}")


def _forward(model_name: str, model: nn.Module, x: torch.Tensor, adj: torch.Tensor | None, edge_index: torch.Tensor | None) -> torch.Tensor:
    if model_name in {"gcn", "graphsage"}:
        return model(x, adj)  # type: ignore[misc]
    return model(x, edge_index)  # type: ignore[misc]


def train_gnn(model_name: str, config: dict[str, Any], seed: int) -> dict[str, Any]:
    set_seed(seed)
    graph = load_graph(config["processed_path"])
    train_cfg = config.get("training", {})
    device = resolve_device(config.get("device", "auto"))
    x = graph["x"].to(device)
    y = graph["y"].float().to(device)
    num_nodes = x.size(0)
    edge_index = prepare_edges(
        graph["edge_index"].to(device),
        num_nodes,
        bidirectional=bool(train_cfg.get("bidirectional_edges", True)),
        self_loops=True,
    )
    adj = None
    if model_name == "gcn":
        adj = normalized_sparse_adj(edge_index, num_nodes, "gcn").to(device)
    elif model_name == "graphsage":
        adj = normalized_sparse_adj(edge_index, num_nodes, "mean").to(device)

    masks = {name: graph[f"{name}_mask"].to(device).bool() for name in ["train", "val", "test"]}
    if bool(train_cfg.get("exclude_unknown_from_graph", False)):
        labelled = (graph["y"].to(device) >= 0)
        keep = labelled[edge_index[0]] & labelled[edge_index[1]]
        edge_index = edge_index[:, keep]
        if model_name == "gcn":
            adj = normalized_sparse_adj(edge_index, num_nodes, "gcn").to(device)
        elif model_name == "graphsage":
            adj = normalized_sparse_adj(edge_index, num_nodes, "mean").to(device)
    train_y = y[masks["train"]]
    pos = int((train_y == 1).sum().item())
    neg = int((train_y == 0).sum().item())
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    model = _build_model(
        model_name,
        in_channels=x.size(1),
        hidden=int(train_cfg.get("hidden_channels", 32)),
        dropout=float(train_cfg.get("dropout", 0.35)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.01)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    max_epochs = int(train_cfg.get("max_epochs", 60))
    patience = int(train_cfg.get("patience", 12))
    best_state = copy.deepcopy(model.state_dict())
    best_val = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model_name, model, x, adj, edge_index)
        loss = loss_fn(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = _forward(model_name, model, x, adj, edge_index)
            probs_val = torch.sigmoid(logits_eval[masks["val"]]).detach().cpu().numpy()
            y_val = y[masks["val"]].detach().cpu().numpy().astype(int)
            val_metrics = binary_metrics(y_val, probs_val)
            val_pr = float(val_metrics["pr_auc"])
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_pr_auc": val_pr})
        if val_pr > best_val:
            best_val = val_pr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    infer_started = time.perf_counter()
    with torch.no_grad():
        logits = _forward(model_name, model, x, adj, edge_index)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    inference_seconds = time.perf_counter() - infer_started

    y_np = graph["y"].numpy()
    mask_np = {name: graph[f"{name}_mask"].numpy().astype(bool) for name in ["train", "val", "test"]}
    metrics = {split: binary_metrics(y_np[mask_np[split]], probs[mask_np[split]]) for split in ["train", "val", "test"]}
    run_tag = str(config.get("run_tag", "")).strip()
    suffix = f"_{run_tag}" if run_tag else ""
    run_id = f"{config.get('experiment_name', 'elliptic_temporal_v1')}_{model_name}_{graph['feature_mode']}{suffix}_seed{seed}"
    output_dir = Path(config.get("output_dir", "outputs"))
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "metrics")
    torch.save({"state_dict": best_state, "config": config, "feature_names": graph["feature_names"]}, output_dir / "checkpoints" / f"{run_id}.pt")
    pd.DataFrame(
        {
            "tx_id": graph["tx_id"],
            "time_step": graph["time_step"].numpy(),
            "y_true": y_np,
            "split": np.select([mask_np["train"], mask_np["val"], mask_np["test"]], ["train", "val", "test"], default="unlabeled_context"),
            "prob_illicit": probs,
            "pred_label": (probs >= 0.5).astype(int),
        }
    ).to_csv(output_dir / "predictions" / f"{run_id}_predictions.csv", index=False)
    result = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "feature_mode": graph["feature_mode"],
        "run_tag": run_tag,
        "split_version": graph["metadata"]["split_version"],
        "edge_count_used": int(edge_index.size(1)),
        "best_epoch": best_epoch,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "runtime": {"train_seconds": train_seconds, "inference_seconds": inference_seconds},
        "device": device_report(device),
        "training": train_cfg,
        "history": history,
        "metrics": metrics,
    }
    write_json(result, output_dir / "metrics" / f"{run_id}_metrics.json")
    return result

