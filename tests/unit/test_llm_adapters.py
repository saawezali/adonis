"""LLM adapter tests — fully offline via httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from adonis.llm import anthropic, openai
from adonis.llm.client import get_client, parse_json_response


def _with_mock(client, handler):
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


# --- anthropic --------------------------------------------------------------


def test_anthropic_complete_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        body = httpx._models.jsonlib.loads(request.content)
        seen["body"] = body
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "hello world"}]},
        )

    client = _with_mock(anthropic.AnthropicClient(model="m-test", api_key="k"), handler)
    out = client.complete("sys", "usr")
    assert out == "hello world"
    assert seen["url"].endswith("/v1/messages")
    assert seen["key"] == "k"
    assert seen["version"] == anthropic.ANTHROPIC_VERSION
    assert seen["body"]["messages"] == [{"role": "user", "content": "usr"}]
    assert seen["body"]["system"] == "sys"


def test_anthropic_complete_json_parses_fenced():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": '```json\n{"ok": true}\n```'}]})

    client = _with_mock(anthropic.AnthropicClient(model="m", api_key="k"), handler)
    assert client.complete_json("sys", "usr") == {"ok": True}


def test_anthropic_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _with_mock(anthropic.AnthropicClient(model="m", api_key="k"), handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("sys", "usr")


# --- openai -----------------------------------------------------------------


def test_openai_complete_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = httpx._models.jsonlib.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = _with_mock(openai.OpenAIClient(model="m-test", api_key="k"), handler)
    assert client.complete("sys", "usr") == "hi"
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_openai_complete_json_uses_json_mode():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["rf"] = httpx._models.jsonlib.loads(request.content).get("response_format")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete_json("sys", "usr") == {"ok": 1}
    assert seen["rf"] == {"type": "json_object"}


def test_openai_explicitly_disables_streaming():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx._models.jsonlib.loads(request.content)
        seen["stream"] = body.get("stream")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete("sys", "usr") == "hi"
    assert seen["stream"] is False


def test_openai_handles_sse_streaming_response():
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":'
            '[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":'
            '[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":'
            '[{"index":0,"delta":{"content":"lo ok"},"finish_reason":null}]}\n\n'
            'data: [DONE]\n\n'
        )
        return httpx.Response(200, content=chunks.encode(), headers={"content-type": "text/event-stream"})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete("sys", "usr") == "hello ok"


def test_openai_sse_detected_by_data_prefix():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = (
            'data: {"choices":[{"delta":{"content":"ping"}}]}\n\n'
            'data: [DONE]\n\n'
        )
        return httpx.Response(200, content=payload.encode())  # no content-type header

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete("sys", "usr") == "ping"


def test_openai_sse_empty_stream_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='data: [DONE]\n\n')

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete("sys", "usr") == ""


def test_openai_non_json_body_raises_with_context():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway hiccup</html>")

    client = _with_mock(openai.OpenAIClient(model="custom-7b", api_key="k"), handler)
    with pytest.raises(RuntimeError, match="non-JSON") as exc_info:
        client.complete("sys", "usr")
    assert "custom-7b" in str(exc_info.value)
    assert "<html>gateway hiccup</html>" in str(exc_info.value)


def test_openai_empty_200_body_raises_with_context():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    with pytest.raises(RuntimeError, match="non-JSON .*<empty body>"):
        client.complete("sys", "usr")


def test_openai_missing_choices_raises_context():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "model overloaded"}})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    with pytest.raises(RuntimeError, match="unexpected payload.*model overloaded"):
        client.complete("sys", "usr")


def test_openai_json_mode_rejected_retries_without_it():
    seen = {"calls": 0, "rf": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["calls"] += 1
        body = httpx._models.jsonlib.loads(request.content)
        seen["rf"].append(body.get("response_format"))
        if seen["calls"] == 1:  # first attempt: gateway rejects response_format
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_object unsupported"}},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    assert client.complete_json("sys", "usr") == {"ok": 1}
    assert seen["calls"] == 2
    assert seen["rf"] == [{"type": "json_object"}, None]


def test_openai_other_4xx_not_retried():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = _with_mock(openai.OpenAIClient(model="m", api_key="k"), handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete_json("sys", "usr")


# --- shared json parsing ----------------------------------------------------


def test_parse_json_bare():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    assert parse_json_response("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_with_prose():
    text = 'Here you go:\n{"a": [1, 2, {"b": "c"}]}\nHope that helps.'
    assert parse_json_response(text) == {"a": [1, 2, {"b": "c"}]}


def test_parse_json_garbage_raises():
    with pytest.raises(ValueError):
        parse_json_response("no json here")


def test_get_client_rejects_bad_tier(monkeypatch):
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "k")
    from adonis import config as cfg
    cfg._settings = None
    with pytest.raises(ValueError):
        get_client("nope")
    cfg._settings = None