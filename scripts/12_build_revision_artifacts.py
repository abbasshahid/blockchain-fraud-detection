"""Build every table and figure the revised manuscript depends on.

Each step is independent and failure-tolerant: a step whose inputs are missing
(for example the LLM runs, which need an API key) is reported as skipped rather
than aborting the rest.  Run it after training and, if available, after report
generation.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockchain_fraud.analysis.ablation_tables import build_ablation_tables  # noqa: E402
from blockchain_fraud.analysis.calibration import build_calibration_artifacts  # noqa: E402
from blockchain_fraud.analysis.case_study import build_case_study  # noqa: E402
from blockchain_fraud.analysis.error_analysis import build_error_artifacts  # noqa: E402
from blockchain_fraud.analysis.explanation_summary import build_explanation_summary  # noqa: E402
from blockchain_fraud.analysis.extra_paper_plots import build_extra_paper_plots  # noqa: E402
from blockchain_fraud.analysis.graph_diagnostics import build_graph_diagnostics  # noqa: E402
from blockchain_fraud.analysis.make_plots import build_plots  # noqa: E402
from blockchain_fraud.analysis.make_tables import build_tables  # noqa: E402
from blockchain_fraud.analysis.revision_plots import build_revision_plots  # noqa: E402
from blockchain_fraud.analysis.significance import build_significance_artifacts  # noqa: E402
from blockchain_fraud.analysis.threshold_analysis import build_threshold_artifacts  # noqa: E402
from blockchain_fraud.blockchain.risk_profile import build_btrep  # noqa: E402
from blockchain_fraud.explain.evidence_value import build_evidence_value_artifacts  # noqa: E402
from blockchain_fraud.explain.faithfulness import build_faithfulness_artifacts  # noqa: E402

PROCESSED = "data/processed/elliptic_temporal_v1_all_original_features.pt"
REFERENCE_RUN = "elliptic_temporal_v1_graphsage_all_original_features_seed11"
STRATIFIED_SUFFIX = "_btrep_v2_stratified"


def build_component_ablation(output_dir: str) -> None:
    """Quantify each BTREP evidence group's contribution.

    Needs a profile sampled across risk deciles, not the top-risk investigation
    sample: on the latter the detector's logit is almost constant, so no group
    could be shown to track it. The profile is built without a checkpoint so
    that feature salience holds the model-agnostic proxy, which keeps the
    ablation from crediting a group derived from the detector itself.
    """
    out = Path(output_dir)
    evidence = out / "explanations" / f"{REFERENCE_RUN}{STRATIFIED_SUFFIX}.json"
    if not evidence.exists():
        build_btrep(
            PROCESSED,
            out / "predictions" / f"{REFERENCE_RUN}_predictions.csv",
            output_dir,
            limit=400,
            selection="stratified",
            output_suffix=STRATIFIED_SUFFIX,
        )
    build_evidence_value_artifacts(evidence, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build revision tables and figures.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-faithfulness", action="store_true", help="Faithfulness needs a saved GNN checkpoint.")
    parser.add_argument("--verbose-errors", action="store_true")
    args = parser.parse_args()
    out = args.output_dir

    steps: list[tuple[str, object]] = [
        ("headline tables", lambda: build_tables(out)),
        ("headline plots", lambda: build_plots(out)),
        ("ranking/confusion/training figures", lambda: build_extra_paper_plots(out)),
        ("significance tests", lambda: build_significance_artifacts(out)),
        ("threshold sensitivity", lambda: build_threshold_artifacts(out)),
        ("calibration", lambda: build_calibration_artifacts(out)),
        ("error analysis", lambda: build_error_artifacts(out)),
        ("graph diagnostics", lambda: build_graph_diagnostics(out)),
        ("retraining ablations", lambda: build_ablation_tables(out)),
    ]
    if not args.skip_faithfulness:
        steps.append(("explanation faithfulness", lambda: build_faithfulness_artifacts(out)))
    steps.extend(
        [
            ("BTREP component ablation", lambda: build_component_ablation(out)),
            ("explanation validation summary", lambda: build_explanation_summary(out)),
            ("case study", lambda: build_case_study(out)),
            ("revision figures", lambda: build_revision_plots(out)),
        ]
    )

    built, skipped = [], []
    for name, fn in steps:
        try:
            fn()  # type: ignore[operator]
            built.append(name)
            print(f"[ok]      {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a missing optional input must not abort the build
            skipped.append((name, str(exc)))
            print(f"[skipped] {name}: {exc}", flush=True)
            if args.verbose_errors:
                traceback.print_exc()

    print(f"\nBuilt {len(built)} artefact groups, skipped {len(skipped)}.")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
