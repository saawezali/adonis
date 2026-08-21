"""Deterministic offline verifier for the entailment pass.

Mirrors DemoJudge: parses the entailment prompt back into span + claim and
scores support lexically (verbatim => 1.0, else normalized fuzzy ratio).
This exercises the plumbing offline; the real semantic check happens with
the configured LLM (--llm).
"""

from __future__ import annotations

import json
import re

from adonis.llm.client import LLMClient
from adonis.verify.span_match import span_match

_FIELD_RE = re.compile(
    r"<cited_span>\n(?P<span>.*?)\n</cited_span>\n"
    r"<claim>\n(?P<claim>.*)\n</claim>",
    re.DOTALL,
)


class DemoVerifier(LLMClient):
    """An LLMClient that decides entailment deterministically from the prompt."""

    model = "demo-verifier"

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return json.dumps(self.complete_json(system, user))

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        del system
        match = _FIELD_RE.match(user)
        if match is None:
            return {"entailed": False, "score": 0.0, "reason": "parse error"}
        span_text = match.group("span").strip()
        claim_text = match.group("claim").strip()
        result = span_match(claim_text, span_text)
        score = 1.0 if result.verbatim else result.fuzzy_ratio / 100.0
        return {
            "entailed": score >= 0.9,
            "score": score,
            "reason": f"demo verifier: fuzzy={result.fuzzy_ratio:.0f}",
        }