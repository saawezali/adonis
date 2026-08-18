"""OpenAI Chat Completions adapter (httpx)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from adonis.llm.client import parse_json_response

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _model_for_tier(tier: str) -> str:
    from adonis.config import get_settings

    settings = get_settings()
    return settings.judge_model if tier == "judge" else settings.extractor_model


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
        content = response.json()["choices"][0]["message"]["content"]
        return str(content)

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
        text = self._chat(
            system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
        )
        return parse_json_response(text)


def build_client(tier: str) -> OpenAIClient:
    from adonis.config import get_settings

    settings = get_settings()
    return OpenAIClient(
        model=_model_for_tier(tier),
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or DEFAULT_BASE_URL,
    )