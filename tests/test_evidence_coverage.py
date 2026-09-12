"""Tests for the evidence-coverage and citation-validity definitions.

These pin down the metric the manuscript reports, so a later change to BTREP
cannot silently move the number.
"""

from __future__ import annotations

import pytest

from blockchain_fraud.explain.validators import available_evidence_ids, validate_report


def _package(*ids: str) -> dict:
    return {
        "transaction_id": "tx",
        "evidence_version": "btrep_test",
        "evidence": {f"group_{i}": {"id": evidence_id} for i, evidence_id in enumerate(ids)},
    }


def _report(cited: list[str], reason_ids: list[list[str]] | None = None) -> dict:
    reasons = [{"claim": f"claim {i}", "evidence_ids": ids} for i, ids in enumerate(reason_ids or [cited])]
    return {
        "transaction_id": "tx",
        "method": "grounded_llm",
        "summary": "summary",
        "reasons": reasons,
        "limitations": ["limited"],
        "cited_evidence_ids": cited,
    }


def test_available_ids_are_read_from_the_package():
    assert available_evidence_ids(_package("pred_1", "struct_1")) == {"pred_1", "struct_1"}


def test_coverage_is_cited_over_supplied():
    package = _package("pred_1", "struct_1", "temp_1", "risk_1")
    result = validate_report(_report(["pred_1", "struct_1"]), package)
    assert result["evidence_coverage"] == pytest.approx(0.5)
    assert result["citation_validity"] == pytest.approx(1.0)
    assert result["valid"]


def test_coverage_denominator_follows_the_btrep_version():
    """A v2 package has one more group, so the same citations cover less of it."""
    v1 = _package(*[f"id_{i}" for i in range(7)])
    v2 = _package(*[f"id_{i}" for i in range(8)])
    cited = _report(["id_0", "id_1", "id_2"])
    assert validate_report(cited, v1)["evidence_coverage"] == pytest.approx(3 / 7)
    assert validate_report(cited, v2)["evidence_coverage"] == pytest.approx(3 / 8)


def test_ids_cited_only_inside_a_reason_still_count():
    package = _package("pred_1", "struct_1")
    result = validate_report(_report([], [["struct_1"]]), package)
    assert result["cited_id_count"] == 1
    assert result["evidence_coverage"] == pytest.approx(0.5)


def test_invented_identifier_lowers_citation_validity_and_invalidates():
    package = _package("pred_1", "struct_1")
    result = validate_report(_report(["pred_1", "cf_1"]), package)
    assert result["invalid_evidence_ids"] == ["cf_1"]
    assert result["citation_validity"] == pytest.approx(0.5)
    assert not result["valid"]


def test_reason_without_any_citation_is_an_unsupported_claim():
    package = _package("pred_1", "struct_1")
    result = validate_report(_report(["pred_1"], [["pred_1"], []]), package)
    assert result["unsupported_claim_count"] == 1
    assert not result["valid"]


def test_wrong_transaction_is_rejected():
    package = _package("pred_1")
    report = _report(["pred_1"])
    report["transaction_id"] = "other"
    result = validate_report(report, package)
    assert not result["transaction_match"]
    assert not result["valid"]
