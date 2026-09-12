from __future__ import annotations

import json
from typing import Any


REPORT_JSON_CONTRACT = """Return only valid compact JSON with no markdown fences, no commentary, and no trailing commas. Use at most 3 reasons. Shape:
{
  "transaction_id": "string",
  "method": "grounded_llm or unconstrained_llm",
  "summary": "one concise paragraph",
  "reasons": [
    {"claim": "supported factual claim", "evidence_ids": ["pred_1", "struct_1"]}
  ],
  "limitations": ["short limitation"],
  "cited_evidence_ids": ["pred_1", "struct_1"]
}
"""


def grounded_prompt(evidence: dict[str, Any]) -> tuple[str, str]:
    instructions = f"""You write concise blockchain fraud investigation reports from supplied model evidence.
You must not invent wallet identities, crimes, external facts, hidden labels, real-world entities, or law-enforcement conclusions.
Every factual reason must cite evidence IDs that appear in the input package.
Use cautious language: model risk, transaction-flow pattern, visible graph neighborhood.
Do not mention ground-truth labels.
{REPORT_JSON_CONTRACT}"""
    user = "Evidence package:\n" + json.dumps(evidence, indent=2, sort_keys=True)
    return instructions, user


def unconstrained_prompt(evidence: dict[str, Any]) -> tuple[str, str]:
    compact = {
        "transaction_id": evidence["transaction_id"],
        "prediction": evidence["prediction"],
        "calibrated_probability": evidence["calibrated_probability"],
        "time_step": evidence["time_step"],
        "structural": evidence["evidence"]["structural"],
        "risk_exposure": evidence["evidence"]["risk_exposure"],
        "motifs": evidence["evidence"]["motifs"]["values"],
    }
    instructions = f"""Write a concise analyst-style blockchain risk report.
You may use natural language freely, but do not claim verified criminal identity or legal guilt.
Return JSON only.
{REPORT_JSON_CONTRACT}"""
    user = "Transaction summary:\n" + json.dumps(compact, indent=2, sort_keys=True)
    return instructions, user
