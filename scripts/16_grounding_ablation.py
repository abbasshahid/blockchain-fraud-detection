"""Controlled test of the evidence-grounding instruction.

The question is whether *instructing* a model to ground every claim in the
supplied evidence changes what it cites. Answering it requires holding
everything else fixed, so for each transaction this script sends the **same
evidence package** to the **same model** twice, varying only the instruction.

Pairs are blocked by model: a transaction is handled end to end by one model,
and models are assigned to disjoint slices. Blocking keeps every comparison
within a model while showing the effect is not an artefact of one of them, and
it spreads the work across per-model request quotas.

Usage::

    python scripts/16_grounding_ablation.py --pairs-per-model 9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from blockchain_fraud.config import load_yaml  # noqa: E402
from blockchain_fraud.explain.llm_client import GeminiGenerateContentClient  # noqa: E402
from blockchain_fraud.explain.prompts import grounded_prompt, unconstrained_prompt  # noqa: E402
from blockchain_fraud.explain.validators import validate_report  # noqa: E402
from blockchain_fraud.utils.io import read_json, write_json, write_table  # noqa: E402

DEFAULT_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
ARMS = {"grounded": grounded_prompt, "unconstrained": unconstrained_prompt}


def _generate(client, prompt_fn, item, arm):
    instructions, user = prompt_fn(item)
    report = client.create_json_report(instructions, user, metadata={"method": arm})
    report["method"] = f"{arm}_llm"
    report.setdefault("transaction_id", str(item["transaction_id"]))
    return report


def run(evidence_path: Path, output_dir: Path, models: list[str], pairs_per_model: int, delay: float) -> pd.DataFrame:
    items = read_json(evidence_path)
    cfg = load_yaml(ROOT / "configs" / "llm.yaml")

    resume_path = output_dir / "explanations" / "grounding_ablation_reports.json"
    done = read_json(resume_path) if resume_path.exists() else []
    seen = {(r["transaction_id"], r["arm"]) for r in done}

    rows: list[dict] = list(done)
    cursor = 0
    for model in models:
        client = GeminiGenerateContentClient(dict(cfg, model=model))
        for item in items[cursor : cursor + pairs_per_model]:
            tx = str(item["transaction_id"])
            for arm, prompt_fn in ARMS.items():
                if (tx, arm) in seen:
                    continue
                try:
                    report = _generate(client, prompt_fn, item, arm)
                except RuntimeError as exc:
                    print(f"  [{model}] {tx} {arm}: {str(exc)[:90]}", flush=True)
                    continue
                validation = validate_report(report, item)
                rows.append(
                    {
                        "transaction_id": tx,
                        "arm": arm,
                        "model": model,
                        "evidence_coverage": validation["evidence_coverage"],
                        "citation_validity": validation["citation_validity"],
                        "valid": validation["valid"],
                        "cited_ids": ",".join(sorted(validation["covered_ids"])),
                        "cited_count": validation["cited_id_count"],
                        "available": validation["available_id_count"],
                        "unsupported_claims": validation["unsupported_claim_count"],
                        "report": report,
                    }
                )
                print(
                    f"  [{model}] {tx} {arm:<13} coverage={validation['evidence_coverage']:.3f} "
                    f"cited={validation['covered_ids']}",
                    flush=True,
                )
                write_json(rows, resume_path)
                time.sleep(delay)
        cursor += pairs_per_model
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, output_dir: Path) -> dict:
    table = df.drop(columns=["report"])
    write_table(table, "table18_grounding_ablation_raw", output_dir / "tables")

    wide = table.pivot_table(index=["transaction_id", "model"], columns="arm", values="evidence_coverage")
    wide = wide.dropna()
    grounded = wide["grounded"].to_numpy(dtype=float)
    unconstrained = wide["unconstrained"].to_numpy(dtype=float)
    diff = grounded - unconstrained

    payload: dict = {
        "design": "paired by transaction, blocked by model; identical evidence package, instruction varied",
        "n_pairs": int(len(diff)),
        "models": sorted(table["model"].unique().tolist()),
        "mean_coverage_grounded": round(float(grounded.mean()), 4),
        "mean_coverage_unconstrained": round(float(unconstrained.mean()), 4),
        "mean_difference": round(float(diff.mean()), 4),
        "citation_validity_grounded": round(float(table.query("arm=='grounded'")["citation_validity"].mean()), 4),
        "citation_validity_unconstrained": round(float(table.query("arm=='unconstrained'")["citation_validity"].mean()), 4),
        "valid_rate_grounded": round(float(table.query("arm=='grounded'")["valid"].mean()), 4),
        "valid_rate_unconstrained": round(float(table.query("arm=='unconstrained'")["valid"].mean()), 4),
        "available_groups": int(table["available"].max()),
    }
    if len(diff) >= 3 and not np.allclose(diff, 0.0):
        w = stats.wilcoxon(grounded, unconstrained, zero_method="wilcox")
        payload["wilcoxon_p"] = float(f"{w.pvalue:.3g}")
        sd = float(diff.std(ddof=1))
        payload["cohens_dz"] = round(float(diff.mean() / sd), 3) if sd > 0 else None
    per_model = {}
    for model, block in wide.groupby(level="model"):
        per_model[model] = {
            "n": int(len(block)),
            "grounded": round(float(block["grounded"].mean()), 4),
            "unconstrained": round(float(block["unconstrained"].mean()), 4),
        }
    payload["per_model"] = per_model
    write_json(payload, output_dir / "metrics" / "grounding_ablation_summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled evidence-grounding ablation.")
    parser.add_argument(
        "--evidence-path",
        default="outputs/explanations/elliptic_temporal_v1_graphsage_all_original_features_seed11_btrep_v2.json",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--pairs-per-model", type=int, default=9)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    df = run(
        Path(args.evidence_path),
        Path(args.output_dir),
        [m.strip() for m in args.models.split(",") if m.strip()],
        args.pairs_per_model,
        args.delay,
    )
    import json

    print(json.dumps(summarise(df, Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
