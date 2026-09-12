from __future__ import annotations

from typing import Any


def render_template_report(evidence: dict[str, Any]) -> dict[str, Any]:
    ev = evidence["evidence"]
    p = ev["prediction"]["probability"]
    risk = ev["risk_exposure"]
    struct = ev["structural"]
    motifs = ev["motifs"]["values"]
    tier = "high" if p >= 0.8 else "elevated" if p >= 0.5 else "lower"
    summary = (
        f"Transaction {evidence['transaction_id']} is assigned {tier} model risk "
        f"with illicit probability {p:.3f}."
    )
    reasons = [
        {
            "claim": f"The visible one-hop neighborhood contains {struct['one_hop_size']} transactions with total degree {struct['total_degree']}.",
            "evidence_ids": ["struct_1"],
        },
        {
            "claim": f"Neighbor model risk has mean {risk['mean_neighbor_risk']:.3f} and maximum {risk['max_neighbor_risk']:.3f}.",
            "evidence_ids": ["risk_1"],
        },
        {
            "claim": f"Detected flow pattern indicators: {', '.join(motifs) if motifs else 'none above threshold'}.",
            "evidence_ids": ["motif_1"],
        },
    ]
    return {
        "transaction_id": evidence["transaction_id"],
        "method": "template",
        "summary": summary,
        "reasons": reasons,
        "limitations": [ev["limitations"]["text"]],
        "cited_evidence_ids": ["pred_1", "struct_1", "risk_1", "motif_1", "limit_1"],
    }

