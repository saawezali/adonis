"""OpenAI Chat Completions adapter (httpx).

Also serves the "custom" provider: any OpenAI-compatible endpoint (Ollama,
vLLM, LM Studio, gateways like OmniRoute/OpenRouter, ...) via base_url.
Not every such server supports `response_format: json_object`; complete_json
falls back to parsing the plain answer when the server rejects json mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from adonis.llm.client import parse_json_response

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _model_for_tier(tier: str) -> str:
    from adonis.config import get_settings

    settings = get_settings()
    return settings.judge_model if tier == "judge" else settings.extractor_model


def _payload_text(response: httpx.Response, limit: int = 300) -> str:
    try:
        body = response.text
    except Exception:  # noqa: BLE001 - defensive read
        body = "<unreadable body>"
    return body[:limit] if body else "<empty body>"


@dataclass
class OpenAIClient:
    """Thin Chat Completions client. One instance per tier; callers may reuse."""

    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    _client: httpx.Client = field(default_factory=httpx.Client, repr=False, compare=False)

    def _chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        if _looks_like_sse(response):
            return _parse_sse(response.text)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"endpoint returned non-JSON (status {response.status_code}, "
                f"model {self.model}): {_payload_text(response)!r}"
            ) from exc
        try:
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"endpoint returned an unexpected payload (status "
                f"{response.status_code}, model {self.model}): {_payload_text(response)!r}"
            ) from exc
        return content

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return self._chat(
            system, user, max_tokens=max_tokens, temperature=temperature, json_mode=False
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        try:
            text = self._chat(
                system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
            )
        except httpx.HTTPStatusError as exc:
            if _rejects_json_mode(exc.response):
                # OpenAI-compatible servers that don't support response_format
                # reject it with a 4xx; retry without it and parse the text.
                text = self._chat(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    json_mode=False,
                )
            else:
                raise
        return parse_json_response(text)


def _rejects_json_mode(response: httpx.Response) -> bool:
    if response.status_code not in (400, 422):
        return False
    hint = response.text.lower()
    return "response_format" in hint or "json_object" in hint


def _looks_like_sse(response: httpx.Response) -> bool:
    """Detect a streaming payload even when the server ignored stream: false."""
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/event-stream"):
        return True
    return response.text.lstrip().startswith("data:")


def _parse_sse(text: str) -> str:
    """Concatenate choices[0].delta.content from SSE chat.completion.chunk events."""
    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def build_client(tier: str) -> OpenAIClient:
    from adonis.config import get_settings

    settings = get_settings()
    return OpenAIClient(
        model=_model_for_tier(tier),
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or DEFAULT_BASE_URL,
    )