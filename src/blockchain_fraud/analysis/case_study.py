"""Forensic case study: from BTREP evidence to an investigation report.

Reviewer 2 asked for a worked example showing how BTREP evidence is generated
and then turned into an investigation-oriented report.  This module selects a
representative transaction, assembles its evidence, its generated report and
its counterfactual measurements, and emits both a compact LaTeX block for the
manuscript and a full markdown walk-through for the supplementary material.

The transaction is chosen deterministically: among the profiled transactions it
takes the one with the largest counterfactual dependence on graph context,
because that is the case where the graph model contributes something a
feature-only detector could not, which is what the example is meant to show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_fraud.utils.io import read_json, write_json


def _load_reports(output_dir: Path, stem: str, method: str = "grounded_llm") -> dict[str, dict[str, Any]]:
    candidates = sorted((output_dir / "explanations").glob(f"{stem}_{method}*_reports.json"))
    if not candidates:
        return {}
    reports = read_json(candidates[-1])
    return {str(r["transaction_id"]): r for r in reports}


def select_case(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    def graph_share(item: dict[str, Any]) -> float:
        cf = item["evidence"].get("counterfactual", {})
        value = cf.get("share_of_decision_from_graph_context")
        return float(value) if value is not None else -1.0

    ranked = sorted(evidence_items, key=graph_share, reverse=True)
    return ranked[0] if ranked else evidence_items[0]


def _latex_escape(text: str) -> str:
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                     ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        text = text.replace(old, new)
    return text


def render_latex(case: dict[str, Any], report: dict[str, Any] | None) -> str:
    ev = case["evidence"]
    struct = ev["structural"]
    risk = ev["risk_exposure"]
    cf = ev.get("counterfactual", {})
    motifs = ", ".join(ev["motifs"]["values"]) or "none above threshold"
    tx = str(case["transaction_id"])
    lines = [
        r"\begin{tabular}{@{}p{0.20\linewidth}p{0.76\linewidth}@{}}",
        r"\toprule",
        rf"Transaction & {_latex_escape(tx)} (time step {case['time_step']}), model risk {case['calibrated_probability']:.3f} \\",
        rf"Structure (\texttt{{struct\_1}}) & in-degree {struct['in_degree']}, out-degree {struct['out_degree']}, "
        rf"one-hop {struct['one_hop_size']}, component {struct['component_size']} \\",
        rf"Neighbour risk (\texttt{{risk\_1}}) & mean {risk['mean_neighbor_risk']:.3f}, max {risk['max_neighbor_risk']:.3f}, "
        rf"{risk['high_risk_neighbor_count']} high-risk neighbours \\",
        rf"Motifs (\texttt{{motif\_1}}) & {_latex_escape(motifs)} \\",
    ]
    if cf:
        share = cf.get("share_of_decision_from_graph_context")
        share_text = f"{float(share) * 100:.0f}\\%" if share is not None else "n/a"
        flip = cf.get("minimum_attributed_features_to_flip")
        flip_text = str(flip) if flip is not None else "no flip within budget"
        lines.append(
            rf"Counterfactual (\texttt{{cf\_1}}) & isolating the transaction moves risk to "
            rf"{cf['probability_without_payment_links']:.3f} ({share_text} of the decision comes from graph context); "
            rf"features needed to flip: {flip_text} \\"
        )
    if report:
        summary = _latex_escape(str(report.get("summary", "")).strip())
        cited = ", ".join(sorted({str(i) for i in report.get("cited_evidence_ids", [])}))
        lines.append(rf"Generated report & {summary} \\")
        lines.append(rf"Cited evidence & \texttt{{{_latex_escape(cited)}}} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_markdown(case: dict[str, Any], report: dict[str, Any] | None, faithfulness: pd.DataFrame | None) -> str:
    parts = [
        f"# Supplementary C — Forensic case study: transaction `{case['transaction_id']}`",
        "",
        "This walk-through shows the complete path from a model prediction to a validated",
        "investigation report for a single transaction, using the artefacts the pipeline",
        "actually wrote. Nothing here is hand-edited.",
        "",
        "## C.1 Prediction",
        "",
        f"- Node index: `{case['node_index']}`",
        f"- Time step: {case['time_step']}",
        f"- Model risk: {case['calibrated_probability']:.4f} → predicted **{case['prediction']}**",
        f"- Evidence version: `{case['evidence_version']}`",
        "",
        "## C.2 BTREP evidence package",
        "",
        "```json",
        json.dumps(case["evidence"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    cf = case["evidence"].get("counterfactual")
    if cf:
        share = cf.get("share_of_decision_from_graph_context")
        parts.extend(
            [
                "## C.3 What the counterfactual evidence means",
                "",
                "The counterfactual group is not a description of the evidence; it is the result of",
                "re-running the trained detector under interventions.",
                "",
                f"- Removing every payment link of this transaction moves its risk from "
                f"{case['calibrated_probability']:.4f} to {cf['probability_without_payment_links']:.4f}.",
                f"- That is a logit drop of {cf['logit_drop_without_payment_links']:.2f}"
                + (f", i.e. {float(share) * 100:.0f}% of the decision comes from graph context." if share is not None else "."),
                f"- Neutralising the single most-attributed feature moves risk to "
                f"{cf['probability_without_top_attributed_feature']:.4f}.",
                f"- Minimum number of attributed features that must be neutralised to cross the 0.5 "
                f"threshold: {cf['minimum_attributed_features_to_flip'] or 'not reached within the search budget'}.",
                "",
                "An investigator can therefore read the report as a statement about *which observable",
                "evidence the score depends on*, and check that statement against the model itself.",
                "",
            ]
        )
    if report:
        parts.extend(
            [
                "## C.4 Generated investigation report",
                "",
                "```json",
                json.dumps(report, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    if faithfulness is not None and not faithfulness.empty:
        rows = faithfulness[faithfulness["node_index"] == case["node_index"]]
        if not rows.empty:
            parts.extend(
                [
                    "## C.5 Faithfulness of this explanation",
                    "",
                    "| Attribution | Comprehensiveness | Sufficiency | Edge fidelity | Flip at k |",
                    "|---|---|---|---|---|",
                ]
            )
            for row in rows.itertuples(index=False):
                flip = "—" if pd.isna(row.counterfactual_flip_k) else int(row.counterfactual_flip_k)
                parts.append(
                    f"| `{row.method}` | {row.comprehensiveness:.2f} | {row.sufficiency:.2f} | "
                    f"{row.edge_fidelity:.2f} | {flip} |"
                )
            parts.extend(
                [
                    "",
                    "Values are logit differences. Comprehensiveness is the drop when the cited features",
                    "are neutralised; sufficiency is the residual when only they are kept; edge fidelity is",
                    "the drop when the transaction is isolated from its payment links.",
                    "",
                ]
            )
    parts.extend(
        [
            "## C.6 What this report is not",
            "",
            "The report describes a model's risk assessment over observable transaction-flow evidence.",
            "It is not a finding of criminal activity, an identification of a wallet owner, or a legal",
            "determination. The evidence-ID validator guarantees that every claim points at supplied",
            "evidence; it does not guarantee that the transaction is illicit.",
            "",
        ]
    )
    return "\n".join(parts)


def build_case_study(
    output_dir: str | Path = "outputs",
    evidence_path: str | Path | None = None,
    docs_dir: str | Path = "docs/supplementary",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if evidence_path is None:
        candidates = sorted((output_dir / "explanations").glob("*_btrep_v2.json")) or sorted(
            (output_dir / "explanations").glob("*_btrep.json")
        )
        if not candidates:
            raise FileNotFoundError("No BTREP evidence file found.")
        evidence_path = candidates[-1]
    evidence_path = Path(evidence_path)
    items = read_json(evidence_path)
    case = select_case(items)

    stem = evidence_path.stem
    reports = _load_reports(output_dir, stem)
    report = reports.get(str(case["transaction_id"]))

    per_node_files = sorted((output_dir / "tables").glob("*_faithfulness_per_node.csv"))
    faithfulness = pd.read_csv(per_node_files[-1]) if per_node_files else None

    latex = render_latex(case, report)
    markdown = render_markdown(case, report, faithfulness)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "table18_case_study.tex").write_text(latex, encoding="utf-8")
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "case_study.md").write_text(markdown, encoding="utf-8")
    write_json(
        {"transaction_id": case["transaction_id"], "node_index": case["node_index"], "has_report": report is not None},
        output_dir / "metrics" / "case_study_selection.json",
    )
    return {"case": case, "report": report, "latex": latex}


if __name__ == "__main__":
    result = build_case_study()
    print(result["latex"])
