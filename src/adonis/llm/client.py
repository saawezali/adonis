"""Provider-independent LLM client interface.

Per PLAN.md: provider-independent. Two tiers are configured independently
(extractor: cheap/fast; judge: larger/smarter). Concrete adapters implement
LLMClient; the choice is made by Settings.llm_provider and loaded dynamically
by get_client().
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from adonis.config import get_settings

_TIERS = ("extractor", "judge")


@runtime_checkable
class LLMClient(Protocol):
    """Minimal client contract used across extraction, judging, and entailment."""

    #: Stable identifier for traceability (see llm_calls.model column).
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's response text. Raise on HTTP/parsing failure."""
        ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Return a JSON object. Adapters use structured output where supported;
        otherwise the response is parsed from text (see parse_json_response)."""
        ...


def get_client(tier: str) -> LLMClient:
    """Build a client for the given tier ('extractor' | 'judge').

    The provider module is found via Settings.llm_provider; each adapter must
    implement `build_client(tier: str) -> LLMClient`.
    """
    if tier not in _TIERS:
        raise ValueError(f"unknown tier: {tier!r}; expected one of {_TIERS}")
    import importlib

    settings = get_settings()
    if settings.llm_api_key == "":
        raise RuntimeError(
            "ADONIS_LLM_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    module = importlib.import_module(f"adonis.llm.{settings.llm_provider}")
    build: Callable[[str], LLMClient] = module.build_client
    return build(tier)


def parse_json_response(text: str) -> dict[str, object]:
    """Parse a JSON object out of a model response.

    Handles: bare JSON, ```json fences, and prose around a single JSON object.
    Raises ValueError when no balanced JSON object can be found.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : i + 1])
                if isinstance(parsed, dict):
                    return parsed
                break
    raise ValueError(f"no JSON object in response: {text[:200]!r}")