"""Prompts for the two report-generation modes.

The two modes exist to isolate one variable: whether *instructing* the model to
ground every claim in supplied evidence changes what it cites. They therefore
receive the **identical evidence package** and differ only in the instruction.

An earlier version gave the unconstrained mode a reduced package with most
evidence identifiers stripped, then scored both modes against the full set of
identifiers. That made the comparison circular -- the unconstrained mode could
not cite what it had never been shown -- so the measured gap reflected the
package, not the instruction. The JSON contract below also no longer contains
real identifiers in its example, because those leaked usable IDs to a mode that
was supposed to have none.
"""

from __future__ import annotations

import json
from typing import Any

REPORT_JSON_CONTRACT = """Return only valid compact JSON with no markdown fences, no commentary, and no trailing commas. Use at most 3 reasons. Shape:
{
  "transaction_id": "string",
  "method": "grounded_llm or unconstrained_llm",
  "summary": "one concise paragraph",
  "reasons": [
    {"claim": "supported factual claim", "evidence_ids": ["<evidence id>"]}
  ],
  "limitations": ["short limitation"],
  "cited_evidence_ids": ["<evidence id>"]
}
"""


def _package(evidence: dict[str, Any]) -> str:
    """The evidence package, serialised identically for both modes."""
    return "Evidence package:\n" + json.dumps(evidence, indent=2, sort_keys=True)


def grounded_prompt(evidence: dict[str, Any]) -> tuple[str, str]:
    """Evidence-grounded mode: every factual claim must cite a supplied identifier."""
    instructions = f"""You write concise blockchain fraud investigation reports from supplied model evidence.
You must not invent wallet identities, crimes, external facts, hidden labels, real-world entities, or law-enforcement conclusions.
Every factual reason must cite evidence IDs that appear in the input package, copied exactly as given.
Cite every evidence group that supports your report, not only the first one you use.
Use cautious language: model risk, transaction-flow pattern, visible graph neighborhood.
Do not mention ground-truth labels.
{REPORT_JSON_CONTRACT}"""
    return instructions, _package(evidence)


def unconstrained_prompt(evidence: dict[str, Any]) -> tuple[str, str]:
    """Ablation: the same package, without the obligation to ground claims in it."""
    instructions = f"""Write a concise analyst-style blockchain risk report from the supplied information.
You may use natural language freely and describe the transaction in your own words.
You do not need to reference the identifiers in the input.
Do not claim verified criminal identity or legal guilt.
Return JSON only.
{REPORT_JSON_CONTRACT}"""
    return instructions, _package(evidence)
