"""Anthropic Messages API adapter (httpx)."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from adonis.llm.client import parse_json_response

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

_JSON_NUDGE = " Respond with a single JSON object and nothing else."


def _model_for_tier(tier: str) -> str:
    from adonis.config import get_settings

    settings = get_settings()
    return settings.judge_model if tier == "judge" else settings.extractor_model


@dataclass
class AnthropicClient:
    """Thin Messages API client. One instance per tier; callers may reuse."""

    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    _client: httpx.Client = field(default_factory=httpx.Client, repr=False, compare=False)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        response = self._client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        return "".join(parts)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        # Anthropic has no response_format knob on this endpoint; we nudge in
        # the system prompt and parse. The extraction layer will iteratively
        # retry on parse failure (M2).
        text = self.complete(
            system + _JSON_NUDGE,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return parse_json_response(text)


def build_client(tier: str) -> AnthropicClient:
    from adonis.config import get_settings

    settings = get_settings()
    return AnthropicClient(
        model=_model_for_tier(tier),
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or DEFAULT_BASE_URL,
    )