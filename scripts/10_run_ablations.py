"""Run the revision ablations requested by the reviewers.

Three families of retraining experiments, all reusing the same trainer,
evaluator and seeds as the headline runs:

* ``local``      -- local-only Elliptic features (first 93 columns), so that the
                    contribution of message passing can be separated from the
                    aggregated neighbourhood features already present in the
                    original feature vector (R1).
* ``splits``     -- alternative leakage-aware temporal boundaries, 50/20/30 and
                    70/15/15, to test the stability of the reported ordering (R2).
* ``nounknown``  -- unknown-labelled nodes removed from message passing, to
                    quantify how much the unlabelled context contributes (R2).

Usage::

    python scripts/10_run_ablations.py --family all --seeds 11,22,33,44,55
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockchain_fraud.config import load_yaml  # noqa: E402
from blockchain_fraud.data.preprocess import preprocess  # noqa: E402
from blockchain_fraud.logging_utils import configure_logging  # noqa: E402
from blockchain_fraud.models.xgboost_baseline import train_xgboost  # noqa: E402
from blockchain_fraud.training.trainer import train_gnn  # noqa: E402
from blockchain_fraud.utils.io import read_json  # noqa: E402

MODELS = ["xgboost", "gcn", "graphsage", "gat"]
GNN_MODELS = ["gcn", "graphsage", "gat"]


def _model_config(model: str, processed_path: str, experiment_name: str, run_tag: str = "") -> dict:
    cfg = load_yaml(f"configs/{model}.yaml")
    cfg["processed_path"] = processed_path
    cfg["experiment_name"] = experiment_name
    if run_tag:
        cfg["run_tag"] = run_tag
    return cfg


def _already_done(output_dir: Path, run_id: str) -> bool:
    path = output_dir / "metrics" / f"{run_id}_metrics.json"
    if not path.exists():
        return False
    try:
        return "metrics" in read_json(path)
    except Exception:
        return False


def _train(model: str, cfg: dict, seed: int) -> dict:
    return train_xgboost(cfg, seed) if model == "xgboost" else train_gnn(model, cfg, seed)


def _run_group(
    label: str,
    models: list[str],
    processed_path: str,
    experiment_name: str,
    feature_mode: str,
    seeds: list[int],
    run_tag: str,
    output_dir: Path,
    resume: bool,
) -> None:
    suffix = f"_{run_tag}" if run_tag else ""
    for seed in seeds:
        for model in models:
            run_id = f"{experiment_name}_{model}_{feature_mode}{suffix}_seed{seed}"
            if resume and _already_done(output_dir, run_id):
                print(f"[{label}] skip {run_id} (already complete)", flush=True)
                continue
            started = time.perf_counter()
            cfg = _model_config(model, processed_path, experiment_name, run_tag)
            result = _train(model, cfg, seed)
            print(
                f"[{label}] {result['run_id']} "
                f"F1={result['metrics']['test']['f1_illicit']:.4f} "
                f"PR-AUC={result['metrics']['test']['pr_auc']:.4f} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )


def run_local_only(seeds: list[int], output_dir: Path, resume: bool) -> None:
    manifest = preprocess(load_yaml("configs/data_local_only.yaml"))
    print(f"[local] processed graph: {manifest['processed_graph']}", flush=True)
    _run_group(
        "local",
        MODELS,
        manifest["processed_graph"],
        "elliptic_temporal_v1",
        "local_only",
        seeds,
        "",
        output_dir,
        resume,
    )


def run_alternative_splits(seeds: list[int], output_dir: Path, resume: bool) -> None:
    for config_name, label in [("configs/data_split_early.yaml", "split50"), ("configs/data_split_late.yaml", "split70")]:
        cfg = load_yaml(config_name)
        manifest = preprocess(cfg)
        print(f"[{label}] processed graph: {manifest['processed_graph']}", flush=True)
        _run_group(
            label,
            MODELS,
            manifest["processed_graph"],
            cfg["experiment_name"],
            cfg["feature_mode"],
            seeds,
            "",
            output_dir,
            resume,
        )


def run_no_unknown(seeds: list[int], output_dir: Path, resume: bool) -> None:
    processed_path = "data/processed/elliptic_temporal_v1_all_original_features.pt"
    suffix = "_nounknown"
    for seed in seeds:
        for model in GNN_MODELS:
            run_id = f"elliptic_temporal_v1_{model}_all_original_features{suffix}_seed{seed}"
            if resume and _already_done(output_dir, run_id):
                print(f"[nounknown] skip {run_id} (already complete)", flush=True)
                continue
            cfg = _model_config(model, processed_path, "elliptic_temporal_v1", "nounknown")
            cfg.setdefault("training", {})["exclude_unknown_from_graph"] = True
            started = time.perf_counter()
            result = train_gnn(model, cfg, seed)
            print(
                f"[nounknown] {result['run_id']} "
                f"F1={result['metrics']['test']['f1_illicit']:.4f} "
                f"PR-AUC={result['metrics']['test']['pr_auc']:.4f} "
                f"edges={result['edge_count_used']} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-requested retraining ablations.")
    parser.add_argument("--family", default="all", choices=["all", "local", "splits", "nounknown"])
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-resume", action="store_true", help="Retrain runs even if metrics already exist.")
    args = parser.parse_args(argv)

    configure_logging()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    resume = not args.no_resume

    started = time.perf_counter()
    if args.family in {"all", "local"}:
        run_local_only(seeds, output_dir, resume)
    if args.family in {"all", "splits"}:
        run_alternative_splits(seeds, output_dir, resume)
    if args.family in {"all", "nounknown"}:
        run_no_unknown(seeds, output_dir, resume)
    print(f"Ablations finished in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
