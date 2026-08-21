"""Shared offline fakes for tests: an LLM client, a NER model, an embedder."""

from __future__ import annotations

import json

import numpy as np


class FakeLLMClient:
    """Canned complete_json responses; records calls for assertions."""

    model = "fake-extractor"

    def __init__(self, by_text: dict[str, dict[str, object]] | None = None) -> None:
        self.by_text = by_text or {}
        self.calls: list[str] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return json.dumps(self.complete_json(system, user, max_tokens=max_tokens))

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        self.calls.append(user)
        for text, response in self.by_text.items():
            if text in user:
                return response
        return {"claims": []}


class FakeNER:
    """Scripted predict_entities; keys are matched as substrings of the text."""

    def __init__(self, by_text: dict[str, list[dict[str, object]]] | None = None) -> None:
        self.by_text = by_text or {}

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[dict[str, object]]:
        for key, mentions in self.by_text.items():
            if key in text:
                return mentions
        return []


class FakeEmbedder:
    """Deterministic hash-based vectors; explicit overrides win."""

    def __init__(self, dim: int = 8, overrides: dict[str, list[float]] | None = None) -> None:
        self.dim = dim
        self.overrides = overrides or {}

    def encode(self, texts: list[str], *, normalize_embeddings: bool = False) -> np.ndarray:
        vecs = []
        for text in texts:
            if text in self.overrides:
                v = np.asarray(self.overrides[text], dtype=np.float32)
            else:
                v = np.asarray(
                    [((hash(text) >> (8 * i)) % 997) / 997.0 for i in range(self.dim)],
                    dtype=np.float32,
                )
                v -= v.mean()
            vecs.append(v)
        out = np.stack(vecs)
        if normalize_embeddings:
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out
