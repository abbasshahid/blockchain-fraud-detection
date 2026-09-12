from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class OpenAIResponsesClient:
    def __init__(self, config: dict[str, Any]):
        self.model = config.get("model", "gpt-4.1-mini")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 700))
        self.timeout = int(config.get("timeout_seconds", 60))
        self.store = bool(config.get("store", False))
        self.api_key = _load_api_key(config)
        if not self.api_key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or configure api_key_file.")

    def create_json_report(self, instructions: str, user_input: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
            "metadata": metadata or {},
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
        text = _extract_output_text(body)
        return _parse_json_text(text)


class GeminiGenerateContentClient:
    def __init__(self, config: dict[str, Any]):
        self.model = config.get("model", "gemini-3.5-flash")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 700))
        self.timeout = int(config.get("timeout_seconds", 60))
        self.api_key = _load_api_key(config, env_name="GEMINI_API_KEY", file_key="gemini_api_key")
        if not self.api_key:
            raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or configure gemini_api_key in api_key_file.")

    def create_json_report(self, instructions: str, user_input: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
        del metadata
        payload = {
            "system_instruction": {"parts": [{"text": instructions}]},
            "contents": [{"parts": [{"text": user_input}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseFormat": {
                    "text": {
                        "mimeType": "APPLICATION_JSON",
                        "schema": _report_json_schema(),
                    }
                },
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
        text = _extract_gemini_text(body)
        return _parse_json_text(text)


class OpenRouterChatClient:
    def __init__(self, config: dict[str, Any]):
        self.model = config.get("model", "google/gemini-2.5-flash-lite")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 1200))
        self.timeout = int(config.get("timeout_seconds", 60))
        self.site_url = str(config.get("site_url", "http://localhost")).strip()
        self.app_title = str(config.get("app_title", "blockchain-fraud-xai")).strip()
        self.api_key = (
            _load_api_key(config, env_name="OPENROUTER_API_KEY", file_key="openrouter_api_key")
            or _load_api_key(config, env_name="OPENROUTER_API_KEY", file_key="openRouter_api_key")
        )
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not found. Set OPENROUTER_API_KEY or configure openRouter_api_key in api_key_file.")

    def create_json_report(self, instructions: str, user_input: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "metadata": metadata or {},
        }
        if bool(metadata and metadata.get("json_schema") == "enabled"):
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_title,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter API HTTP {exc.code}: {detail}") from exc
        text = _extract_openrouter_text(body)
        return _parse_json_text(text)


def _load_api_key(config: dict[str, Any], env_name: str = "OPENAI_API_KEY", file_key: str = "openai_api_key") -> str:
    env_key = os.environ.get(env_name, "").strip()
    if env_key:
        return env_key
    path = config.get("api_key_file")
    if not path:
        return ""
    key_path = Path(path)
    if not key_path.exists():
        return ""
    data = json.loads(key_path.read_text(encoding="utf-8"))
    return str(data.get(file_key, "")).strip()


def _extract_output_text(response: dict[str, Any]) -> str:
    if "output_text" in response and response["output_text"]:
        return str(response["output_text"])
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                parts.append(str(content["text"]))
    if not parts:
        raise RuntimeError(f"No output text found in OpenAI response id={response.get('id')}")
    return "\n".join(parts)


def _extract_gemini_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                parts.append(str(part["text"]))
    if not parts:
        raise RuntimeError("No text content found in Gemini response.")
    return "\n".join(parts)


def _extract_openrouter_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("No choices found in OpenRouter response.")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        content = "\n".join(p for p in parts if p)
    if not content:
        raise RuntimeError("No message content found in OpenRouter response.")
    return str(content)


def _report_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "transaction_id": {"type": "string"},
            "method": {"type": "string"},
            "summary": {"type": "string"},
            "reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["claim", "evidence_ids"],
                },
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
            "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["transaction_id", "method", "summary", "reasons", "limitations", "cited_evidence_ids"],
    }


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    elif cleaned.lower().startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        snippet = cleaned[:500].replace("\n", "\\n")
        raise RuntimeError(f"Could not parse model JSON output. First parse error: {first_error}. Output starts with: {snippet!r}") from first_error


def backoff_sleep(attempt: int) -> None:
    time.sleep(min(8, 2**attempt))
