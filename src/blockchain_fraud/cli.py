from __future__ import annotations

import argparse
from pathlib import Path

from blockchain_fraud.analysis.make_plots import build_plots
from blockchain_fraud.analysis.make_tables import build_tables
from blockchain_fraud.blockchain.risk_profile import build_btrep
from blockchain_fraud.config import load_yaml
from blockchain_fraud.data.preprocess import preprocess
from blockchain_fraud.data.validate_raw import validate_raw
from blockchain_fraud.explain.report_generator import generate_reports
from blockchain_fraud.logging_utils import configure_logging
from blockchain_fraud.models.xgboost_baseline import train_xgboost
from blockchain_fraud.training.trainer import train_gnn


def _load_model_config(model: str, config_path: str | None) -> dict:
    path = config_path or f"configs/{model}.yaml"
    return load_yaml(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="blockchain_fraud")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data")
    p.add_argument("--raw-dir", default="elliptic_bitcoin_dataset")
    p.add_argument("--output-dir", default="outputs")

    p = sub.add_parser("preprocess")
    p.add_argument("--config", default="configs/data.yaml")

    p = sub.add_parser("train")
    p.add_argument("--model", choices=["xgboost", "gcn", "graphsage", "gat"], required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--config")

    p = sub.add_parser("train-all")
    p.add_argument("--models", default="xgboost,gcn,graphsage,gat")
    p.add_argument("--seeds", default="11,22,33,44,55")

    p = sub.add_parser("generate-evidence")
    p.add_argument("--model", default="graphsage")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--processed-path", default="data/processed/elliptic_temporal_v1_all_original_features.pt")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--evidence-version",
        default="v2",
        choices=["v1", "v2"],
        help="v2 adds counterfactual evidence and gradient salience by re-running the checkpoint.",
    )

    p = sub.add_parser("generate-reports")
    p.add_argument("--evidence-path")
    p.add_argument("--method", default="template", choices=["template", "grounded_llm", "unconstrained_llm"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--llm-config", default="configs/llm.yaml")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("build-paper-artifacts")
    p.add_argument("--output-dir", default="outputs")

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "validate-data":
        report = validate_raw(args.raw_dir, args.output_dir)
        print(f"Validated {report['counts']['nodes']} nodes and {report['counts']['edges']} edges.")
    elif args.command == "preprocess":
        manifest = preprocess(load_yaml(args.config))
        print(f"Saved processed graph: {manifest['processed_graph']}")
    elif args.command == "train":
        cfg = _load_model_config(args.model, args.config)
        result = train_xgboost(cfg, args.seed) if args.model == "xgboost" else train_gnn(args.model, cfg, args.seed)
        print(f"Finished {result['run_id']} test PR-AUC={result['metrics']['test']['pr_auc']:.4f}")
    elif args.command == "train-all":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        for seed in seeds:
            for model in models:
                cfg = _load_model_config(model, None)
                result = train_xgboost(cfg, seed) if model == "xgboost" else train_gnn(model, cfg, seed)
                print(f"Finished {result['run_id']} test F1={result['metrics']['test']['f1_illicit']:.4f}")
    elif args.command == "generate-evidence":
        run_id = f"elliptic_temporal_v1_{args.model}_all_original_features_seed{args.seed}"
        pred_path = Path(args.output_dir) / "predictions" / f"{run_id}_predictions.csv"
        use_v2 = args.evidence_version == "v2"
        evidence = build_btrep(
            args.processed_path,
            pred_path,
            args.output_dir,
            args.limit,
            model_name=args.model if use_v2 else None,
            checkpoint_path=(Path(args.output_dir) / "checkpoints" / f"{run_id}.pt") if use_v2 else None,
            output_suffix="_btrep_v2" if use_v2 else "_btrep",
        )
        version = evidence[0]["evidence_version"] if evidence else "n/a"
        print(f"Generated {len(evidence)} BTREP evidence records ({version}).")
    elif args.command == "generate-reports":
        evidence_path = args.evidence_path
        if evidence_path is None:
            candidates = sorted((Path(args.output_dir) / "explanations").glob("*_btrep.json"))
            if not candidates:
                raise FileNotFoundError("No BTREP evidence file found. Run generate-evidence first.")
            evidence_path = str(candidates[-1])
        reports = generate_reports(evidence_path, args.method, args.output_dir, args.llm_config, args.limit)
        print(f"Generated {len(reports)} reports.")
    elif args.command == "build-paper-artifacts":
        tables = build_tables(args.output_dir)
        plots = build_plots(args.output_dir)
        print(f"Built {len(tables)} tables and {len(plots)} figures.")


if __name__ == "__main__":
    main()
