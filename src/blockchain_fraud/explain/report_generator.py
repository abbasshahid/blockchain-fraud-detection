from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from blockchain_fraud.config import load_yaml
from blockchain_fraud.explain.llm_client import GeminiGenerateContentClient, OpenAIResponsesClient, OpenRouterChatClient, backoff_sleep
from blockchain_fraud.explain.prompts import grounded_prompt, unconstrained_prompt
from blockchain_fraud.explain.template_explainer import render_template_report
from blockchain_fraud.explain.validators import validate_report
from blockchain_fraud.utils.io import read_json, write_json


def generate_reports(
    evidence_path: str | Path,
    method: str = "template",
    output_dir: str | Path = "outputs",
    llm_config_path: str | Path = "configs/llm.yaml",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    evidence_items = read_json(evidence_path)
    if limit is not None:
        evidence_items = evidence_items[:limit]
    reports: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    llm_config = load_yaml(llm_config_path) if method in {"grounded_llm", "unconstrained_llm"} else {}
    client = _build_llm_client(llm_config) if method in {"grounded_llm", "unconstrained_llm"} else None
    max_retries = int(llm_config.get("max_retries", 1)) if llm_config else 0
    api_max_retries = int(llm_config.get("api_max_retries", 4)) if llm_config else 0
    delay_between_requests = float(llm_config.get("delay_between_requests_seconds", 0.0)) if llm_config else 0.0
    stem = Path(evidence_path).stem
    output_tag = _output_tag(llm_config, method)
    reports_path = Path(output_dir) / "explanations" / f"{stem}_{method}_reports.json"
    validations_path = Path(output_dir) / "metrics" / f"{stem}_{method}_report_validation.json"
    if output_tag:
        reports_path = Path(output_dir) / "explanations" / f"{stem}_{method}_{output_tag}_reports.json"
        validations_path = Path(output_dir) / "metrics" / f"{stem}_{method}_{output_tag}_report_validation.json"
    if reports_path.exists():
        reports = read_json(reports_path)
    if validations_path.exists():
        validations = read_json(validations_path)
    completed = {str(report.get("transaction_id")) for report in reports}
    for item in evidence_items:
        if str(item["transaction_id"]) in completed:
            continue
        if method == "template":
            report = render_template_report(item)
        elif method in {"grounded_llm", "unconstrained_llm"}:
            assert client is not None
            prompt_fn = grounded_prompt if method == "grounded_llm" else unconstrained_prompt
            instructions, user_input = prompt_fn(item)
            report = _call_with_validation_retry(client, instructions, user_input, item, method, max_retries, api_max_retries)
        else:
            raise ValueError(f"Unknown report method: {method}")
        reports.append(report)
        validations.append({"transaction_id": item["transaction_id"], **validate_report(report, item)})
        write_json(reports, reports_path)
        write_json(validations, validations_path)
        if delay_between_requests > 0 and method in {"grounded_llm", "unconstrained_llm"}:
            time.sleep(delay_between_requests)
    write_json(reports, reports_path)
    write_json(validations, validations_path)
    return reports


def _build_llm_client(config: dict[str, Any]) -> OpenAIResponsesClient | GeminiGenerateContentClient | OpenRouterChatClient:
    provider = str(config.get("provider", "openai_responses")).lower()
    if provider == "openai_responses":
        return OpenAIResponsesClient(config)
    if provider in {"google_gemini", "gemini"}:
        return GeminiGenerateContentClient(config)
    if provider == "openrouter":
        return OpenRouterChatClient(config)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _output_tag(config: dict[str, Any], method: str) -> str:
    if method not in {"grounded_llm", "unconstrained_llm"}:
        return ""
    tag = str(config.get("output_tag", "")).strip()
    if tag:
        return tag
    provider = str(config.get("provider", "")).lower().strip()
    return provider if provider and provider not in {"google_gemini", "gemini"} else ""


def _call_with_validation_retry(
    client: OpenAIResponsesClient,
    instructions: str,
    user_input: str,
    evidence: dict[str, Any],
    method: str,
    max_retries: int,
    api_max_retries: int,
) -> dict[str, Any]:
    last_report: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        report = _call_api_with_retries(client, instructions, user_input, evidence, method, api_max_retries)
        report["method"] = method
        last_report = report
        validation = validate_report(report, evidence)
        if validation["valid"]:
            return report
        user_input = (
            user_input
            + "\n\nPrevious JSON failed validation. Fix only these issues and return JSON only: "
            + str(validation)
        )
        backoff_sleep(attempt)
    assert last_report is not None
    return last_report


def _call_api_with_retries(
    client: OpenAIResponsesClient | GeminiGenerateContentClient | OpenRouterChatClient,
    instructions: str,
    user_input: str,
    evidence: dict[str, Any],
    method: str,
    api_max_retries: int,
) -> dict[str, Any]:
    for attempt in range(api_max_retries + 1):
        try:
            return client.create_json_report(
                instructions,
                user_input,
                metadata={"method": method, "transaction_id": str(evidence["transaction_id"])[:64]},
            )
        except RuntimeError as exc:
            text = str(exc)
            transient = any(marker in text for marker in ["HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "UNAVAILABLE"])
            if not transient or attempt >= api_max_retries:
                raise
            backoff_sleep(attempt)
    raise RuntimeError("Unreachable API retry state.")
