from __future__ import annotations

from pydantic import BaseModel, Field


class ReportReason(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)


class InvestigationReport(BaseModel):
    transaction_id: str
    method: str
    summary: str
    reasons: list[ReportReason]
    limitations: list[str]
    cited_evidence_ids: list[str]

