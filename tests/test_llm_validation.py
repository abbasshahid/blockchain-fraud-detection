from blockchain_fraud.explain.validators import validate_report


def test_invalid_evidence_id_fails():
    evidence = {
        "transaction_id": "tx",
        "evidence": {
            "prediction": {"id": "pred_1"},
            "limitations": {"id": "limit_1"},
        },
    }
    report = {
        "transaction_id": "tx",
        "method": "template",
        "summary": "summary",
        "reasons": [{"claim": "claim", "evidence_ids": ["missing"]}],
        "limitations": ["limited"],
        "cited_evidence_ids": ["pred_1", "missing"],
    }
    result = validate_report(report, evidence)
    assert not result["valid"]
    assert result["invalid_evidence_ids"] == ["missing"]

