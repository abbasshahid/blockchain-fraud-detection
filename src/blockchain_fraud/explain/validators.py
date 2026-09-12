"""Deterministic validation of a generated investigation report.

Reviewer 1 asked for the evidence-coverage metric to be defined precisely and
for citation validity to be kept distinct from explanation faithfulness.  This
module computes only the *citation-level* properties; faithfulness is measured
separately in :mod:`blockchain_fraud.explain.faithfulness` by perturbing the
detector.

Let ``A`` be the set of evidence identifiers present in the BTREP package
supplied to the generator, and ``C`` the set of identifiers cited anywhere in
the report (in ``cited_evidence_ids`` or in any reason's ``evidence_ids``).
Then

    citation validity  =  |C \\ (C \\ A)| / |C|   =  |C n A| / |C|
    evidence coverage  =  |C n A| / |A|

Citation validity is 1.0 when the report invents no identifier; evidence
coverage is the share of the supplied evidence groups the report actually used.
Coverage is derived from the package itself rather than from a hard-coded list,
so BTREP v1 (7 evidence groups) and BTREP v2 (8, after the counterfactual group
was added) are each scored against their own denominator.

A report is ``valid`` when it cites no unavailable identifier, addresses the
correct transaction, and supports every reason with at least one identifier.
"""

from __future__ import annotations

from typing import Any

from blockchain_fraud.explain.schemas import InvestigationReport


def available_evidence_ids(evidence: dict[str, Any]) -> set[str]:
    """Every evidence identifier present in a BTREP package, any version."""
    return {v["id"] for v in evidence["evidence"].values() if isinstance(v, dict) and "id" in v}


def validate_report(report: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    parsed = InvestigationReport.model_validate(report)
    available = available_evidence_ids(evidence)
    cited = set(parsed.cited_evidence_ids)
    for reason in parsed.reasons:
        cited.update(reason.evidence_ids)
    invalid = sorted(cited - available)
    covered = sorted(cited & available)

    unsupported = [reason.claim for reason in parsed.reasons if not reason.evidence_ids]
    transaction_match = parsed.transaction_id == evidence["transaction_id"]
    citation_validity = 1.0 if not cited else len(covered) / len(cited)
    return {
        "valid": (not invalid) and transaction_match and not unsupported,
        "invalid_evidence_ids": invalid,
        "transaction_match": transaction_match,
        "citation_validity": citation_validity,
        "evidence_coverage": len(covered) / max(1, len(available)),
        "covered_ids": covered,
        "available_id_count": len(available),
        "cited_id_count": len(cited),
        "reason_count": len(parsed.reasons),
        "unsupported_claim_count": len(unsupported),
        "unsupported_claims": unsupported,
        "evidence_version": evidence.get("evidence_version", "unknown"),
    }
