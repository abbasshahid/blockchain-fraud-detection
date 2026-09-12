from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from blockchain_fraud.training.evaluator import binary_metrics
from blockchain_fraud.utils.io import ensure_dir, load_graph, write_json


def train_xgboost(config: dict[str, Any], seed: int) -> dict[str, Any]:
    graph = load_graph(config["processed_path"])
    x = graph["x"].numpy()
    y = graph["y"].numpy()
    masks = {name: graph[f"{name}_mask"].numpy().astype(bool) for name in ["train", "val", "test"]}
    params_cfg = config.get("params", {})
    train_idx = masks["train"] & (y >= 0)
    val_idx = masks["val"] & (y >= 0)
    test_idx = masks["test"] & (y >= 0)

    neg = int((y[train_idx] == 0).sum())
    pos = int((y[train_idx] == 1).sum())
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": int(params_cfg.get("max_depth", 6)),
        "eta": float(params_cfg.get("learning_rate", 0.05)),
        "subsample": float(params_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(params_cfg.get("colsample_bytree", 0.8)),
        "min_child_weight": float(params_cfg.get("min_child_weight", 1)),
        "lambda": float(params_cfg.get("reg_lambda", 1.0)),
        "alpha": float(params_cfg.get("reg_alpha", 0.0)),
        "scale_pos_weight": neg / max(1, pos),
        "seed": seed,
        "tree_method": "hist",
        "nthread": -1,
    }
    dtrain = xgb.DMatrix(x[train_idx], label=y[train_idx], feature_names=graph["feature_names"])
    dval = xgb.DMatrix(x[val_idx], label=y[val_idx], feature_names=graph["feature_names"])
    dall = xgb.DMatrix(x, feature_names=graph["feature_names"])

    started = time.perf_counter()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(params_cfg.get("n_estimators", 500)),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(params_cfg.get("early_stopping_rounds", 50)),
        verbose_eval=False,
    )
    train_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    probs = booster.predict(dall, iteration_range=(0, booster.best_iteration + 1))
    inference_seconds = time.perf_counter() - infer_started

    metrics = {
        split: binary_metrics(y[masks[split] & (y >= 0)], probs[masks[split] & (y >= 0)])
        for split in ["train", "val", "test"]
    }
    run_tag = str(config.get("run_tag", "")).strip()
    suffix = f"_{run_tag}" if run_tag else ""
    run_id = f"{config.get('experiment_name', 'elliptic_temporal_v1')}_xgboost_{graph['feature_mode']}{suffix}_seed{seed}"
    output_dir = Path(config.get("output_dir", "outputs"))
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "metrics")
    booster.save_model(output_dir / "checkpoints" / f"{run_id}.json")
    pred_df = pd.DataFrame(
        {
            "tx_id": graph["tx_id"],
            "time_step": graph["time_step"].numpy(),
            "y_true": y,
            "split": np.select([masks["train"], masks["val"], masks["test"]], ["train", "val", "test"], default="unlabeled_context"),
            "prob_illicit": probs,
            "pred_label": (probs >= 0.5).astype(int),
        }
    )
    pred_df.to_csv(output_dir / "predictions" / f"{run_id}_predictions.csv", index=False)
    result = {
        "run_id": run_id,
        "model": "xgboost",
        "seed": seed,
        "feature_mode": graph["feature_mode"],
        "run_tag": run_tag,
        "split_version": graph["metadata"]["split_version"],
        "best_iteration": int(booster.best_iteration),
        "runtime": {"train_seconds": train_seconds, "inference_seconds": inference_seconds},
        "params": params,
        "metrics": metrics,
    }
    write_json(result, output_dir / "metrics" / f"{run_id}_metrics.json")
    return result

