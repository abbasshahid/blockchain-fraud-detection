"""Export representative generated reports for the supplementary material.

Reviewer 1 asked that representative generated reports be published alongside
the prompts and the schema.  This writes a readable document containing the
evidence package and the report for a small, deterministic selection of
transactions, together with their validation results.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N_EXAMPLES = 3


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    explanations = ROOT / "outputs" / "explanations"
    metrics = ROOT / "outputs" / "metrics"

    evidence_files = sorted(explanations.glob("*_btrep_v2.json")) or sorted(explanations.glob("*_btrep.json"))
    if not evidence_files:
        raise SystemExit("No BTREP evidence found; run generate-evidence first.")
    evidence = {str(item["transaction_id"]): item for item in _load(evidence_files[-1])}

    lines = [
        "# Supplementary B — Representative generated reports",
        "",
        "Every report below was produced by the pipeline at temperature 0 and validated",
        "automatically. Nothing is hand-edited or selected for quality: the examples are the",
        f"first {N_EXAMPLES} transactions of each run, in the order the pipeline processed them.",
        "",
        "See Supplementary A for the prompts, the JSON schema and the validation procedure.",
        "",
    ]

    for report_file in sorted(explanations.glob("*_reports.json")):
        reports = _load(report_file)
        if not reports:
            continue
        stem = report_file.stem.replace("_reports", "")
        validation_file = metrics / f"{stem}_report_validation.json"
        validations = {}
        if validation_file.exists():
            validations = {str(v["transaction_id"]): v for v in _load(validation_file)}

        lines.extend([f"## `{report_file.name}`", "", f"Reports in this run: **{len(reports)}**", ""])
        if validations:
            valid = sum(1 for v in validations.values() if v.get("valid"))
            coverage = [v["evidence_coverage"] for v in validations.values() if v.get("evidence_coverage") is not None]
            lines.append(
                f"Validated: {valid}/{len(validations)} valid"
                + (f", mean evidence coverage {sum(coverage) / len(coverage):.4f}" if coverage else "")
            )
            lines.append("")

        for report in reports[:N_EXAMPLES]:
            tx = str(report["transaction_id"])
            lines.extend([f"### Transaction `{tx}`", ""])
            if tx in evidence:
                lines.extend(
                    [
                        "Evidence package supplied to the generator:",
                        "",
                        "```json",
                        json.dumps(evidence[tx]["evidence"], indent=2, sort_keys=True),
                        "```",
                        "",
                    ]
                )
            lines.extend(["Generated report:", "", "```json", json.dumps(report, indent=2, sort_keys=True), "```", ""])
            if tx in validations:
                lines.extend(["Validation:", "", "```json", json.dumps(validations[tx], indent=2, sort_keys=True), "```", ""])

    destination = ROOT / "docs" / "supplementary" / "example_reports.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
