"""FastAPI console: provider setup + API key + pipeline + report access.

Routes:
  GET  /                        settings page (static HTML)
  GET  /report                  -> /reports/index.html
  GET  /reports/*               statically served report artifacts
  GET  /api/settings            current config (key masked)
  POST /api/settings            persist config to .env (blank = keep)
  POST /api/test                probe a provider endpoint (no save)
  POST /api/pipeline/run        run the M4 pipeline (+ render report)
  GET  /api/status              store + provider summary
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adonis.config import (
    ALL_PROVIDERS,
    PROVIDER_DEFAULTS,
    Settings,
    get_settings,
    mask_secret,
    provider_for_tier,
    reload_settings,
    save_settings,
)
from adonis.db import apply_migrations, get_conn
from adonis.llm.client import LLMClient
from adonis.pipeline import run, wipe_pipeline_rows

_PAGE = Path(__file__).parent / "static" / "settings.html"

_ENV_KEYS = {
    "llm_provider": "ADONIS_LLM_PROVIDER",
    "extractor_provider": "ADONIS_EXTRACTOR_PROVIDER",
    "judge_provider": "ADONIS_JUDGE_PROVIDER",
    "llm_api_key": "ADONIS_LLM_API_KEY",
    "llm_base_url": "ADONIS_LLM_BASE_URL",
    "extractor_model": "ADONIS_EXTRACTOR_MODEL",
    "judge_model": "ADONIS_JUDGE_MODEL",
}
_PROVIDER_FIELDS = ("llm_provider", "extractor_provider", "judge_provider")


class SettingsPatch(BaseModel):
    llm_provider: str = ""
    extractor_provider: str = ""
    judge_provider: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    extractor_model: str = ""
    judge_model: str = ""


class PipelineRequest(BaseModel):
    use_llm: bool = False
    refresh: bool = False


def _check_providers(values: dict[str, str]) -> dict[str, str]:
    for field_name in _PROVIDER_FIELDS:
        value = values.get(field_name, "")
        if value and value not in ALL_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}: {value!r} not in {ALL_PROVIDERS}",
            )
    return values


def _public(s: Settings | None = None) -> dict[str, object]:
    settings = s or get_settings()
    return {
        "providers": list(ALL_PROVIDERS),
        "defaults": PROVIDER_DEFAULTS,
        "llm_provider": settings.llm_provider,
        "extractor_provider": settings.extractor_provider or "",
        "judge_provider": settings.judge_provider or "",
        "extractor_effective": provider_for_tier(settings, "extractor"),
        "judge_effective": provider_for_tier(settings, "judge"),
        "llm_api_key": mask_secret(settings.llm_api_key),
        "key_set": bool(settings.llm_api_key),
        "llm_base_url": settings.llm_base_url or "",
        "extractor_model": settings.extractor_model,
        "judge_model": settings.judge_model,
    }


def _probe(
    provider: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    tier: Literal["extractor", "judge"],
) -> dict[str, object]:
    from adonis.llm.anthropic import DEFAULT_BASE_URL as ANTHROPIC_URL
    from adonis.llm.anthropic import AnthropicClient
    from adonis.llm.openai import DEFAULT_BASE_URL as OPENAI_URL
    from adonis.llm.openai import OpenAIClient

    settings = get_settings()
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    key = api_key or settings.llm_api_key
    model = model or defaults.get(f"{tier}_model") or getattr(settings, f"{tier}_model")
    client: LLMClient

    if provider == "anthropic":
        url = base_url or ANTHROPIC_URL
        client = AnthropicClient(model=model, api_key=key, base_url=url)
    elif provider in ("openai", "custom"):
        url = base_url or settings.llm_base_url or OPENAI_URL
        if provider == "custom" and not (base_url or settings.llm_base_url):
            raise HTTPException(
                status_code=400,
                detail="provider 'custom' needs a base URL (Ollama/vLLM/LM Studio endpoint)",
            )
        client = OpenAIClient(model=model, api_key=key, base_url=url)
    else:
        raise HTTPException(status_code=400, detail=f"unknown provider {provider!r}")

    started = time.monotonic()
    try:
        response = client.complete(
            "You are a connectivity probe. Answer briefly.",
            "Say ok if you can read this message.",
            max_tokens=16,
        )
    except Exception as exc:  # surface any adapter error to the UI
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    reply = str(response).strip()
    return {
        "ok": bool(reply),
        "latency_ms": int((time.monotonic() - started) * 1000),
        "provider": provider,
        "model": client.model,
        "base_url": url,
        "reply": reply[:120],
    }


def create_app() -> FastAPI:
    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Adonis console", version="0.1.0")
    app.mount(
        "/reports",
        StaticFiles(directory=str(settings.reports_dir)),
        name="reports",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_PAGE.read_text(encoding="utf-8"))

    @app.get("/report")
    def report_page() -> RedirectResponse:
        return RedirectResponse("/reports/index.html")

    @app.get("/api/settings")
    def api_settings() -> dict[str, object]:
        return _public()

    @app.post("/api/settings")
    def api_save(patch: SettingsPatch) -> dict[str, object]:
        values = _check_providers(patch.model_dump())
        env_patch = {
            _ENV_KEYS[k]: str(v).strip()
            for k, v in values.items()
            if v is not None
        }
        env_patch = {k: v for k, v in env_patch.items() if v}
        path = save_settings(env_patch)
        reload_settings()
        return {**_public(), "env_file": str(path.resolve())}

    @app.post("/api/test")
    def api_test(patch: SettingsPatch | None = None) -> dict[str, object]:
        values = _check_providers(patch.model_dump()) if patch else {}
        settings = get_settings()
        provider = values.get("llm_provider") or settings.llm_provider
        return _probe(
            provider,
            api_key=values.get("llm_api_key", ""),
            base_url=values.get("llm_base_url", ""),
            model=values.get("judge_model", ""),
            tier="judge",
        )

    @app.post("/api/pipeline/run")
    def api_pipeline(req: PipelineRequest) -> dict[str, object]:
        from adonis.judge.demo import DemoJudge
        from adonis.pair.embed import load_embedder
        from adonis.report.render import render_report
        from adonis.verify.demo import DemoVerifier

        if req.use_llm:
            from adonis.llm.client import get_client

            client = get_client("judge")
            verifier = get_client("judge")
        else:
            client = DemoJudge()
            verifier = DemoVerifier()

        apply_migrations()
        conn = get_conn()
        try:
            if req.refresh:
                wipe_pipeline_rows(conn)
            embedder = load_embedder()
            stats = run(conn, client, embedder, verifier)
            render_report(conn, get_settings().reports_dir / "index.html")
        finally:
            conn.close()

        data = dataclasses.asdict(stats)
        data["citation_faithfulness"] = stats.citation_faithfulness
        data["judge_model"] = client.model
        data["judge_provider"] = provider_for_tier(get_settings(), "judge")
        return data

    @app.get("/api/status")
    def api_status() -> dict[str, object]:
        apply_migrations()
        conn = get_conn()

        def _count(table: str) -> int:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row is not None else 0

        try:
            counts = {
                "documents": _count("documents"),
                "claims": _count("claims"),
                "candidate_pairs": _count("candidate_pairs"),
                "judge_outputs": _count("judge_outputs"),
                "flags": _count("flags"),
                "eval_labels": _count("eval_labels"),
            }
        finally:
            conn.close()
        return {**counts, **_public()}

    return app


app = create_app()