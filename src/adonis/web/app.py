"""FastAPI console: provider setup + API key + pipeline + report + documents + flags + jobs.

Routes (legacy):
  GET  /                        app shell (new) — legacy settings at /settings.html
  GET  /report                  -> /reports/index.html
  GET  /reports/*               statically served report artifacts
  GET  /api/settings            current config (key masked)
  POST /api/settings            persist config to .env (blank = keep)
  POST /api/test                probe a provider endpoint (no save)
  POST /api/pipeline/run        run the M4 pipeline (+ render report)
  GET  /api/status              store + provider summary

Routes (new console):
  GET  /api/documents           ?q=&source=&limit=&offset=  paginated list
  GET  /api/documents/{id}      document detail + claims preview
  GET  /api/documents/{id}/text raw_text
  POST /api/documents/upload    multipart files auto-ingest
  DELETE /api/documents/{id}    hard-delete DB row only
  GET  /api/claims              ?document_id=&limit=&offset=
  GET  /api/flags               ?label=&min_conf=&q=&limit=&offset=&verified=
  PATCH /api/flags/{id}         {user_decision, notes}
  GET  /api/eval                eval report json
  GET  /api/eval/status         labeled counts
  POST /api/ingest              {path?} ingest corpus_dir
  POST /api/extract             {document_id?, use_llm?}
  GET  /api/jobs                list jobs
  POST /api/jobs                create pipeline job {use_llm, refresh, ...}
  GET  /api/jobs/{id}           job status
  GET  /api/labels/pending      ?pool=&limit=
  POST /api/labels              {claim_a_id, claim_b_id, label, notes} -> staged (pending)
  GET  /api/labels/staged       ?status=&limit=&offset=
  POST /api/labels/staged/{id}/review {action: approve|reject, notes?}
  GET  /api/connections         list
  POST /api/connections         {kind, name, config}
  POST /api/connections/{id}/sync
  DELETE /api/connections/{id}
  GET  /api/connections/drive/auth
  GET  /api/connections/drive/callback
  POST /api/connections/drive/sync
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
from adonis.db import (
    apply_migrations,
    delete_document,
    get_conn,
    insert_staged_label,
    list_claims_for_document,
    list_documents,
)

# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------
_PAGE_LEGACY = Path(__file__).parent / "static" / "settings.html"
_PAGE_APP = Path(__file__).parent / "static" / "app.html"

_ENV_KEYS = {
    "llm_provider": "ADONIS_LLM_PROVIDER",
    "extractor_provider": "ADONIS_EXTRACTOR_PROVIDER",
    "judge_provider": "ADONIS_JUDGE_PROVIDER",
    "llm_api_key": "ADONIS_LLM_API_KEY",
    "llm_base_url": "ADONIS_LLM_BASE_URL",
    "extractor_model": "ADONIS_EXTRACTOR_MODEL",
    "judge_model": "ADONIS_JUDGE_MODEL",
    "google_client_id": "ADONIS_GOOGLE_CLIENT_ID",
    "google_client_secret": "ADONIS_GOOGLE_CLIENT_SECRET",
    "google_redirect_uri": "ADONIS_GOOGLE_REDIRECT_URI",
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
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""


class PipelineRequest(BaseModel):
    use_llm: bool = False
    refresh: bool = False


class FlagPatch(BaseModel):
    user_decision: str | None = None  # confirmed | rejected | unsure | null
    notes: str | None = None


class ExtractRequest(BaseModel):
    document_id: str | None = None
    limit: int | None = None
    use_llm: bool = False


class ConnectionCreate(BaseModel):
    kind: str  # local | notion | drive
    name: str = ""
    config: dict = {}  # type: ignore[type-arg]


class LabelCreate(BaseModel):
    claim_a_id: str
    claim_b_id: str
    label: str
    notes: str | None = None
    labeled_by: str = "web"


class StagedReview(BaseModel):
    action: str  # approve | reject
    reviewed_by: str = "reviewer"
    notes: str | None = None


class JobCreate(BaseModel):
    use_llm: bool = False
    refresh: bool = False
    top_k: int | None = None
    selected_per_claim: int | None = None
    max_pairs: int | None = None


def _check_providers(values: dict[str, str]) -> dict[str, str]:
    for field_name in _PROVIDER_FIELDS:
        value = values.get(field_name, "")
        if value and value not in ALL_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"{field_name}: {value!r} not in {ALL_PROVIDERS}")
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
        "google_client_id": mask_secret(settings.google_client_id or ""),
        "google_configured": bool(settings.google_client_id and settings.google_client_secret),
        "google_redirect_uri": settings.google_redirect_uri or "",
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
    from adonis.llm.client import LLMClient
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
            raise HTTPException(status_code=400, detail="provider 'custom' needs a base URL")
        client = OpenAIClient(model=model, api_key=key, base_url=url)
    else:
        raise HTTPException(status_code=400, detail=f"unknown provider {provider!r}")

    started = time.monotonic()
    try:
        response = client.complete("You are a connectivity probe. Answer briefly.", "Say ok if you can read this message.", max_tokens=16)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    reply = str(response).strip()
    return {"ok": bool(reply), "latency_ms": int((time.monotonic() - started) * 1000), "provider": provider, "model": client.model, "base_url": url, "reply": reply[:120]}


# ---------------------------------------------------------------------------
# Jobs — async via threads + DB polling
# ---------------------------------------------------------------------------

def _run_job_thread(job_id: str, params: dict[str, object]) -> None:
    """Background thread entry for pipeline jobs. Updates jobs table."""
    from adonis.db import update_job

    # Re-open connection in thread (sqlite thread safety)
    conn = get_conn()
    try:
        update_job(conn, job_id, status="running", started_at=datetime.now(UTC).isoformat())
    finally:
        conn.close()

    use_llm = bool(params.get("use_llm"))
    refresh = bool(params.get("refresh"))
    top_k = params.get("top_k")
    selected_per_claim = params.get("selected_per_claim")
    max_pairs = params.get("max_pairs")

    try:
        from adonis.judge.demo import DemoJudge
        from adonis.pair.embed import load_embedder
        from adonis.pipeline import run, wipe_pipeline_rows
        from adonis.report.render import render_report
        from adonis.verify.demo import DemoVerifier

        if use_llm:
            from adonis.llm.client import get_client

            client = get_client("judge")
            verifier = get_client("judge")
        else:
            client = DemoJudge()
            verifier = DemoVerifier()

        apply_migrations()
        conn2 = get_conn()
        try:
            if refresh:
                wipe_pipeline_rows(conn2)
            embedder = load_embedder()
            stats = run(conn2, client, embedder, verifier, top_k=top_k, selected_per_claim=selected_per_claim, max_pairs=max_pairs)  # type: ignore[arg-type]
            render_report(conn2, get_settings().reports_dir / "index.html")
            # Persist result
            import dataclasses

            data = dataclasses.asdict(stats)
            data["citation_faithfulness"] = stats.citation_faithfulness
            data["judge_model"] = client.model
            from adonis.config import provider_for_tier as pft

            data["judge_provider"] = pft(get_settings(), "judge")
            conn_j = get_conn()
            try:
                update_job(conn_j, job_id, status="done", finished_at=datetime.now(UTC).isoformat(), result_json=json.dumps(data))
            finally:
                conn_j.close()
        finally:
            conn2.close()
    except Exception as exc:
        conn_e = get_conn()
        try:
            update_job(conn_e, job_id, status="error", finished_at=datetime.now(UTC).isoformat(), error=f"{type(exc).__name__}: {exc}")
        finally:
            conn_e.close()


def create_app() -> FastAPI:
    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Adonis console", version="0.2.0")
    app.mount("/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # Serve new shell if exists else legacy
        page = _PAGE_APP if _PAGE_APP.exists() else _PAGE_LEGACY
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page() -> HTMLResponse:
        return HTMLResponse(_PAGE_LEGACY.read_text(encoding="utf-8"))

    @app.get("/app", response_class=HTMLResponse)
    def app_page() -> HTMLResponse:
        page = _PAGE_APP if _PAGE_APP.exists() else _PAGE_LEGACY
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/report")
    def report_page() -> RedirectResponse:
        return RedirectResponse("/reports/index.html")

    # ------------------------------------------------------------------
    # Settings / status / pipeline (legacy + extended)
    # ------------------------------------------------------------------
    @app.get("/api/settings")
    def api_settings() -> dict[str, object]:
        return _public()

    @app.post("/api/settings")
    def api_save(patch: SettingsPatch) -> dict[str, object]:
        values = _check_providers(patch.model_dump())
        env_patch = {_ENV_KEYS[k]: str(v).strip() for k, v in values.items() if v is not None}
        env_patch = {k: v for k, v in env_patch.items() if v}
        path = save_settings(env_patch)
        reload_settings()
        return {**_public(), "env_file": str(path.resolve())}

    @app.post("/api/test")
    def api_test(patch: SettingsPatch | None = None) -> dict[str, object]:
        values = _check_providers(patch.model_dump()) if patch else {}
        settings = get_settings()
        provider = values.get("llm_provider") or settings.llm_provider
        return _probe(provider, api_key=values.get("llm_api_key", ""), base_url=values.get("llm_base_url", ""), model=values.get("judge_model", ""), tier="judge")

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
                from adonis.pipeline import wipe_pipeline_rows

                wipe_pipeline_rows(conn)
            embedder = load_embedder()
            from adonis.pipeline import run

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
            # staged counts if table exists
            try:
                counts["staged_labels"] = _count("staged_labels")
                counts["staged_pending"] = int(conn.execute("SELECT COUNT(*) FROM staged_labels WHERE status='pending'").fetchone()[0])
            except Exception:
                counts["staged_labels"] = 0
                counts["staged_pending"] = 0
            try:
                counts["connections"] = _count("connections")
                counts["jobs"] = _count("jobs")
            except Exception:
                pass
        finally:
            conn.close()
        return {**counts, **_public()}

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    @app.get("/api/documents")
    def api_list_documents(
        q: str | None = Query(None),
        source: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            rows, total = list_documents(conn, q=q, source=source, limit=limit, offset=offset)
            items = []
            for r in rows:
                # claims count per doc (cheap subquery would be better but ok)
                cc = conn.execute("SELECT COUNT(*) FROM claims WHERE document_id=?", (r["id"],)).fetchone()[0]
                items.append(
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "title": r["title"],
                        "path": r["path"],
                        "format": r["format"],
                        "text_len": r["text_len"],
                        "content_hash": r["content_hash"],
                        "ingested_at": r["ingested_at"],
                        "parse_warnings": json.loads(r["parse_warnings_json"]) if r["parse_warnings_json"] else [],
                        "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                        "claims": int(cc),
                    }
                )
            return {"items": items, "total": total, "limit": limit, "offset": offset}
        finally:
            conn.close()

    @app.get("/api/documents/{doc_id}")
    def api_get_document(doc_id: str) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT id, source, title, path, format, raw_text, metadata_json, content_hash, ingested_at, parse_warnings_json FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="document not found")
            claim_rows, _ = list_claims_for_document(conn, doc_id, limit=5)
            return {
                "id": row["id"],
                "source": row["source"],
                "title": row["title"],
                "path": row["path"],
                "format": row["format"],
                "raw_text": row["raw_text"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                "content_hash": row["content_hash"],
                "ingested_at": row["ingested_at"],
                "parse_warnings": json.loads(row["parse_warnings_json"]) if row["parse_warnings_json"] else [],
                "preview_claims": [
                    {
                        "id": c["id"],
                        "claim_text": c["claim_text"],
                        "span": [c["citation_span_start"], c["citation_span_end"]],
                        "topics": json.loads(c["topics_json"]) if c["topics_json"] else [],
                        "triviality": c["triviality_score"],
                    }
                    for c in claim_rows
                ],
            }
        finally:
            conn.close()

    @app.get("/api/documents/{doc_id}/text")
    def api_get_document_text(doc_id: str) -> JSONResponse:
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT raw_text, title FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="document not found")
            return JSONResponse({"id": doc_id, "title": row["title"], "raw_text": row["raw_text"]})
        finally:
            conn.close()

    @app.delete("/api/documents/{doc_id}")
    def api_delete_document(doc_id: str) -> dict[str, object]:
        """Hard-delete DB row only (not original file on disk)."""
        apply_migrations()
        conn = get_conn()
        try:
            counts = delete_document(conn, doc_id)
            if counts.get("documents", 0) == 0:
                raise HTTPException(status_code=404, detail="document not found")
            return {"deleted": doc_id, "counts": counts}
        finally:
            conn.close()

    @app.post("/api/documents/upload")
    async def api_upload_documents(files: list[UploadFile] = File(...)) -> dict[str, object]:  # noqa: B008
        """Upload one or more files; auto-ingest on arrival. Returns IngestStats-like summary."""
        apply_migrations()
        settings = get_settings()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        # Reject too many files early
        if len(files) > 50:
            raise HTTPException(status_code=413, detail="too many files (max 50)")
        from adonis.ingest.pipeline import ingest_corpus as do_ingest

        conn = get_conn()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            saved: list[Path] = []
            total_bytes = 0
            for uf in files:
                # Stream in chunks to avoid OOM; abort early if over limit
                chunks: list[bytes] = []
                file_bytes = 0
                while True:
                    chunk = await uf.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > max_bytes or total_bytes > max_bytes:
                        raise HTTPException(status_code=413, detail=f"upload exceeds {settings.max_upload_mb} MB limit")
                    chunks.append(chunk)
                content = b"".join(chunks)
                # Preserve extension for ingest routing
                fname = Path(uf.filename or "upload.txt").name
                dest = tmp_path / fname
                # Avoid overwrite collisions
                idx = 1
                while dest.exists():
                    dest = tmp_path / f"{Path(fname).stem}_{idx}{Path(fname).suffix}"
                    idx += 1
                dest.write_bytes(content)
                saved.append(dest)
            # Ingest the temp corpus
            stats = do_ingest(tmp_path, conn)
            stats.doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        try:
            pass
        finally:
            conn.close()
        return {
            "files_seen": stats.files_seen,
            "inserted": stats.inserted,
            "duplicates": stats.duplicates,
            "failed": stats.failed,
            "skipped": stats.skipped,
            "ignored": stats.ignored,
            "documents_in_store": stats.doc_count,
            "errors": stats.errors,
            "notes": stats.notes,
        }

    @app.get("/api/claims")
    def api_list_claims(
        document_id: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            if document_id:
                rows, total = list_claims_for_document(conn, document_id, limit=limit, offset=offset)
                items = [
                    {
                        "id": r["id"],
                        "claim_text": r["claim_text"],
                        "citation_span_start": r["citation_span_start"],
                        "citation_span_end": r["citation_span_end"],
                        "entities": json.loads(r["entities_json"]) if r["entities_json"] else [],
                        "topics": json.loads(r["topics_json"]) if r["topics_json"] else [],
                        "temporal": json.loads(r["temporal_json"]) if r["temporal_json"] else None,
                        "scope": json.loads(r["scope_json"]) if r["scope_json"] else None,
                        "triviality": r["triviality_score"],
                        "extraction_model": r["extraction_model"],
                    }
                    for r in rows
                ]
                return {"items": items, "total": total, "limit": limit, "offset": offset}
            # global listing
            total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            rows = conn.execute(
                "SELECT id, document_id, claim_text, citation_span_start, citation_span_end, entities_json, topics_json, temporal_json, scope_json, triviality_score, extraction_model FROM claims ORDER BY extraction_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            items = [
                {
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "claim_text": r["claim_text"],
                    "citation_span_start": r["citation_span_start"],
                    "citation_span_end": r["citation_span_end"],
                    "entities": json.loads(r["entities_json"]) if r["entities_json"] else [],
                    "topics": json.loads(r["topics_json"]) if r["topics_json"] else [],
                    "temporal": json.loads(r["temporal_json"]) if r["temporal_json"] else None,
                    "scope": json.loads(r["scope_json"]) if r["scope_json"] else None,
                    "triviality": r["triviality_score"],
                    "extraction_model": r["extraction_model"],
                }
                for r in rows
            ]
            return {"items": items, "total": int(total), "limit": limit, "offset": offset}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------
    @app.get("/api/flags")
    def api_list_flags(
        label: str | None = Query(None),
        min_conf: float | None = Query(None, ge=0, le=1),
        verified: str | None = Query(None),
        q: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            where: list[str] = []
            params: list[object] = []
            if label:
                where.append("f.final_label = ?")
                params.append(label)
            if min_conf is not None:
                where.append("f.final_confidence >= ?")
                params.append(min_conf)
            if verified in ("pass", "fail"):
                where.append("vr.overall_pass = ?")
                params.append(1 if verified == "pass" else 0)
            if q:
                where.append("(c1.claim_text LIKE ? OR c2.claim_text LIKE ? OR d1.title LIKE ? OR d2.title LIKE ?)")
                like = f"%{q}%"
                params.extend([like, like, like, like])
            where_sql = f" WHERE {' AND '.join(where)}" if where else ""
            total = conn.execute(
                f"SELECT COUNT(*) FROM flags f JOIN candidate_pairs cp ON cp.id=f.candidate_pair_id JOIN claims c1 ON c1.id=cp.claim_a_id JOIN claims c2 ON c2.id=cp.claim_b_id JOIN documents d1 ON d1.id=c1.document_id JOIN documents d2 ON d2.id=c2.document_id JOIN judge_outputs jo ON jo.candidate_pair_id=cp.id JOIN verification_results vr ON vr.judge_output_id=jo.id{where_sql}",
                tuple(params),
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT f.id AS flag_id, f.final_label, f.final_confidence, f.user_decision, f.notes, c1.claim_text AS text_a, c2.claim_text AS text_b, d1.title AS doc_a_title, d2.title AS doc_b_title, d1.id AS doc_a_id, d2.id AS doc_b_id, d1.raw_text AS raw_a, d2.raw_text AS raw_b, jo.cited_span_a_start, jo.cited_span_a_end, jo.cited_span_b_start, jo.cited_span_b_end, vr.overall_pass, vr.span_a_verbatim, vr.span_b_verbatim, vr.span_a_fuzzy, vr.span_b_fuzzy, vr.span_a_entailment, vr.span_b_entailment, jo.judge_model, jo.prompt_version, jo.reasoning_text FROM flags f JOIN candidate_pairs cp ON cp.id=f.candidate_pair_id JOIN claims c1 ON c1.id=cp.claim_a_id JOIN claims c2 ON c2.id=cp.claim_b_id JOIN documents d1 ON d1.id=c1.document_id JOIN documents d2 ON d2.id=c2.document_id JOIN judge_outputs jo ON jo.candidate_pair_id=cp.id JOIN verification_results vr ON vr.judge_output_id=jo.id{where_sql} ORDER BY f.final_confidence DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            items = [
                {
                    "id": r["flag_id"],
                    "final_label": r["final_label"],
                    "final_confidence": float(r["final_confidence"]),
                    "user_decision": r["user_decision"],
                    "notes": r["notes"],
                    "text_a": r["text_a"],
                    "text_b": r["text_b"],
                    "doc_a_title": r["doc_a_title"],
                    "doc_b_title": r["doc_b_title"],
                    "doc_a_id": r["doc_a_id"],
                    "doc_b_id": r["doc_b_id"],
                    "span_a": [r["cited_span_a_start"], r["cited_span_a_end"]],
                    "span_b": [r["cited_span_b_start"], r["cited_span_b_end"]],
                    "raw_a": r["raw_a"],
                    "raw_b": r["raw_b"],
                    "verification": {
                        "overall_pass": bool(r["overall_pass"]),
                        "span_a_verbatim": bool(r["span_a_verbatim"]),
                        "span_b_verbatim": bool(r["span_b_verbatim"]),
                        "span_a_fuzzy": float(r["span_a_fuzzy"]),
                        "span_b_fuzzy": float(r["span_b_fuzzy"]),
                        "span_a_entailment": float(r["span_a_entailment"]),
                        "span_b_entailment": float(r["span_b_entailment"]),
                    },
                    "judge_model": r["judge_model"],
                    "prompt_version": r["prompt_version"],
                    "reasoning": r["reasoning_text"],
                }
                for r in rows
            ]
            return {"items": items, "total": int(total), "limit": limit, "offset": offset}
        finally:
            conn.close()

    @app.patch("/api/flags/{flag_id}")
    def api_patch_flag(flag_id: str, patch: FlagPatch) -> dict[str, object]:
        allowed = {"confirmed", "rejected", "unsure", None}
        if patch.user_decision not in allowed:
            raise HTTPException(status_code=400, detail="user_decision must be confirmed|rejected|unsure|null")
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM flags WHERE id=?", (flag_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="flag not found")
            now = datetime.now(UTC).isoformat() if patch.user_decision else None
            conn.execute(
                "UPDATE flags SET user_decision=?, user_decision_at=?, notes=COALESCE(?, notes) WHERE id=?",
                (patch.user_decision, now, patch.notes, flag_id),
            )
            conn.commit()
            updated = conn.execute("SELECT id, user_decision, user_decision_at, notes FROM flags WHERE id=?", (flag_id,)).fetchone()
            return {"id": updated["id"], "user_decision": updated["user_decision"], "user_decision_at": updated["user_decision_at"], "notes": updated["notes"]}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    @app.get("/api/eval")
    def api_eval() -> dict[str, object]:
        from adonis.eval.harness import run_eval

        apply_migrations()
        conn = get_conn()
        try:
            report = run_eval(conn)
            return report.as_dict()
        finally:
            conn.close()

    @app.get("/api/eval/status")
    def api_eval_status() -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM eval_labels").fetchone()[0]
            try:
                staged_total = conn.execute("SELECT COUNT(*) FROM staged_labels").fetchone()[0]
                pending = conn.execute("SELECT COUNT(*) FROM staged_labels WHERE status='pending'").fetchone()[0]
            except Exception:
                staged_total = 0
                pending = 0
            per_label = conn.execute("SELECT label, COUNT(*) AS c FROM eval_labels GROUP BY label").fetchall()
            breakdown = {r["label"]: r["c"] for r in per_label}
            return {"total": int(total), "breakdown": breakdown, "staged_total": int(staged_total), "staged_pending": int(pending)}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Ingest / Extract triggers
    # ------------------------------------------------------------------
    @app.post("/api/ingest")
    def api_ingest(payload: dict | None = None) -> dict[str, object]:  # type: ignore[type-arg]
        apply_migrations()
        settings = get_settings()
        raw_path = (payload or {}).get("path") if payload else None
        corpus = Path(raw_path) if raw_path else settings.corpus_dir
        # Sandbox: must be inside corpus_dir (or equal)
        try:
            resolved = corpus.resolve()
            base = settings.corpus_dir.resolve()
            # Allow base itself or any subpath; reject traversal outside
            if resolved != base and base not in resolved.parents:
                raise HTTPException(status_code=403, detail="corpus path outside allowed directory")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="invalid corpus path")
        if not corpus.exists():
            raise HTTPException(status_code=404, detail=f"corpus path not found: {corpus}")
        from adonis.ingest.pipeline import ingest_corpus as do_ingest

        conn = get_conn()
        try:
            stats = do_ingest(corpus, conn)
            stats.doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            return {
                "files_seen": stats.files_seen,
                "inserted": stats.inserted,
                "duplicates": stats.duplicates,
                "failed": stats.failed,
                "skipped": stats.skipped,
                "ignored": stats.ignored,
                "documents_in_store": stats.doc_count,
                "errors": stats.errors,
                "notes": stats.notes,
            }
        finally:
            conn.close()

    @app.post("/api/sample")
    def api_sample() -> dict[str, object]:
        """Generate synthetic sample corpus (MIT) and ingest it — for first-run onboarding."""
        apply_migrations()
        settings = get_settings()
        # Reuse generator logic inline to avoid import cycle
        sample_files = {
            "01_roadmap.md": "# Roadmap — Project Atlas\n\nAtlas ships in March.\n\nAtlas is our flagship product.\n\nThe launch budget is 500 euros.\n\nOnboarding is required for EU customers.\n",
            "02_finance.md": "# Finance — Atlas Budget\n\nAtlas ships in July.\n\nThe launch budget is 600 euros.\n\nOnboarding is required for US customers.\n",
            "03_product.md": "# Product — Atlas Overview\n\nAtlas is our flagship product.\n\nWe are considering a hiring freeze for Q3.\n",
            "04_ops_eu.md": "# Ops — EU Rollout\n\nOnboarding is required for EU customers.\n\nSupport window is 9am–5pm CET.\n",
            "05_ops_us.md": "# Ops — US Rollout\n\nOnboarding is required for US customers.\n\nSupport window is 9am–5pm EST.\n",
        }
        out = settings.corpus_dir / "sample"
        out.mkdir(parents=True, exist_ok=True)
        for name, text in sample_files.items():
            (out / name).write_text(text, encoding="utf-8")
        from adonis.ingest.pipeline import ingest_corpus as do_ingest
        conn = get_conn()
        try:
            stats = do_ingest(out, conn)
            stats.doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            return {
                "sample_dir": str(out),
                "files_seen": stats.files_seen,
                "inserted": stats.inserted,
                "duplicates": stats.duplicates,
                "documents_in_store": stats.doc_count,
                "note": "Synthetic MIT sample — expected demo flags: genuine_contradiction (March vs July) + different_scope (EU vs US)",
            }
        finally:
            conn.close()

    @app.post("/api/extract")
    def api_extract(req: ExtractRequest) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            if req.use_llm:
                from adonis.llm.client import get_client

                client = get_client("extractor")
            else:
                # Lightweight demo client (same behavior as scripts/extract_claims.DemoClient)
                # to avoid importing scripts as a package.
                import json as _json
                import re as _re

                _SENT_RE = _re.compile(r"^([^.!?]*[.!?]|\S[^\n]*)", _re.DOTALL)
                _TEXT_RE = _re.compile(r"<text>\n(.*)\n</text>", _re.DOTALL)

                class _DemoClient:
                    model = "demo-canned"
                    def complete(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
                        return _json.dumps(self.complete_json(system, user))
                    def complete_json(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> dict[str, object]:
                        m = _TEXT_RE.search(user)
                        if m is None:
                            return {"claims": []}
                        chunk_text = m.group(1)
                        sent = _SENT_RE.match(chunk_text)
                        if sent is None:
                            return {"claims": []}
                        claim_text = sent.group(1).strip()
                        if not claim_text or claim_text.endswith(("?", "!")):
                            return {"claims": []}
                        start = chunk_text.find(claim_text)
                        if start < 0:
                            return {"claims": []}
                        return {"claims": [{"claim_text": claim_text, "span_start": start, "span_end": start+len(claim_text), "triviality_score": 0.1, "topics": ["demo"], "temporal": None, "scope": None}]}

                client = _DemoClient()  # type: ignore[no-redef]
            # Reuse the same logic as scripts/extract_claims.run without importing scripts
            from adonis.db import (
                insert_claim,
                insert_entity_mention,
                insert_llm_call,
                iter_documents,
                update_claim_entities,
                upsert_entity,
            )
            _HAS_NER = False
            _HAS_CLAIMS = False
            try:
                from adonis.extract.canonicalize import ClaimMention, cluster_mentions
                from adonis.extract.claims import extract_document_claims, prompt_version
                from adonis.extract.entities import extract_mentions
                _HAS_NER = True
                _HAS_CLAIMS = True
            except Exception:
                try:
                    from adonis.extract.claims import extract_document_claims, prompt_version
                    _HAS_CLAIMS = True
                except Exception:
                    _HAS_CLAIMS = False
                    # Minimal fallback without spacy/gliner
                    import re as _re2
                    def _fallback_chunks(text: str, max_chars: int = 1200):  # type: ignore[no-redef]
                        if not text.strip():
                            return []
                        # naive sentence split
                        sens = _re2.split(r'(?<=[.!?])\s+', text.strip())
                        chunks = []
                        cur = ""; cur_start = 0
                        pos = 0
                        for s in sens:
                            if not s.strip():
                                continue
                            # find s in original text from pos
                            start = text.find(s, pos)
                            if start == -1:
                                start = pos
                            end = start + len(s)
                            pos = end
                            if cur and len(cur) + 1 + len(s) > max_chars:
                                chunks.append((cur_start, cur_start+len(cur), cur))
                                cur = s; cur_start = start
                            else:
                                if cur:
                                    cur += " " + s
                                else:
                                    cur = s; cur_start = start
                        if cur:
                            chunks.append((cur_start, cur_start+len(cur), cur))
                        return chunks
                    def extract_document_claims(client, raw_text: str, max_chars=None, cutoff=None):  # type: ignore[no-redef]
                        # trivial demo extraction: one claim per chunk first sentence
                        import re as _r
                        _SENT = _r.compile(r"^([^.!?]*[.!?]|\S[^\n]*)", _r.DOTALL)
                        chunks = _fallback_chunks(raw_text, max_chars or 1200)
                        from dataclasses import field as _fld2
                        @dataclass
                        class _CR:
                            claim_text: str; span_start: int; span_end: int; topics: list; temporal: dict|None; scope: dict|None; triviality_score: float=0.1
                        @dataclass
                        class _ES:
                            chunks: int=0; llm_calls: int=0; claims_from_llm: int=0; trivial_dropped: int=0; span_dropped: int=0; shape_dropped: int=0; errors: list=_fld2(default_factory=list)
                        claims=[]; es=_ES(chunks=len(chunks), llm_calls=len(chunks))
                        for (cs, ce, txt) in chunks:
                            # use demo client if available else naive
                            try:
                                import re as _re3
                                _TR = _re3.compile(r"<text>\n(.*)\n</text>", _re3.DOTALL)
                                sys_, usr = client.complete_json.__self__ if hasattr(client.complete_json, '__self__') else (None, None)
                            except: pass
                            m = _SENT.match(txt)
                            if not m: continue
                            ct = m.group(1).strip()
                            if not ct or ct.endswith(("?","!")): continue
                            # find offset in raw_text
                            off = raw_text.find(ct, cs)
                            if off==-1: off=cs
                            claims.append(_CR(claim_text=ct, span_start=off, span_end=off+len(ct), topics=["demo"], temporal=None, scope=None))
                        es.claims_from_llm=len(claims)
                        return claims, es
                    def prompt_version(): return "claims_v1-fallback"  # type: ignore[no-redef]
            import time as _time

            # Inline extract run (subset of scripts.extract_claims.run)
            from dataclasses import dataclass
            from dataclasses import field as _field
            from datetime import UTC as _UTC
            from datetime import datetime as _dt

            @dataclass
            class _Stats:
                docs_seen: int = 0; docs_failed: int = 0; chunks: int = 0; llm_calls: int = 0
                claims_llm: int = 0; claims_inserted: int = 0; entities: int = 0; mentions: int = 0
                trivial_dropped: int = 0; span_dropped: int = 0; shape_dropped: int = 0; errors: list[str] = _field(default_factory=list)

            _stats = _Stats()
            _extraction_at = _dt.now(_UTC).isoformat()
            for _doc in iter_documents(conn, document_id=req.document_id, limit=req.limit):
                _stats.docs_seen += 1
                _started = _time.monotonic()
                try:
                    _claims, _es = extract_document_claims(client, _doc["raw_text"])  # type: ignore[arg-type]
                    _stats.chunks += _es.chunks; _stats.llm_calls += _es.llm_calls; _stats.claims_llm += _es.claims_from_llm
                    _stats.trivial_dropped += _es.trivial_dropped; _stats.span_dropped += _es.span_dropped; _stats.shape_dropped += _es.shape_dropped
                    _stats.errors.extend(_es.errors)
                    if _HAS_CLAIMS:
                        try:
                            from adonis.extract.claims import span_text as _span_text
                        except Exception:
                            def _span_text(doc_text: str, cl) -> str:  # type: ignore[no-redef]
                                return doc_text[cl.span_start:cl.span_end]
                    else:
                        def _span_text(doc_text: str, cl) -> str:  # type: ignore[no-redef]
                            return doc_text[cl.span_start:cl.span_end]
                    for _cl in _claims:
                        _cid = insert_claim(conn, document_id=_doc["id"], claim_text=_cl.claim_text, span_start=_cl.span_start, span_end=_cl.span_end, topics=_cl.topics, temporal=_cl.temporal, scope=_cl.scope, triviality_score=_cl.triviality_score, extraction_model=client.model, extraction_at=_extraction_at)
                        _stats.claims_inserted += 1
                        if not _HAS_NER:
                            continue
                        _span_val = _span_text(_doc["raw_text"], _cl)
                        try:
                            _mentions = extract_mentions(_span_val, offset=_cl.span_start)
                        except Exception as _exc:
                            _stats.errors.append(f"document {_doc['id']}: NER failed: {_exc!r}")
                            continue
                        _cms = [ClaimMention(claim_id=_cid, mention=m) for m in _mentions]
                        _clusters = cluster_mentions(_cms)
                        _eids: list[str] = []
                        for _cluster in _clusters:
                            _eid = upsert_entity(conn, canonical_name=_cluster.canonical_name, aliases=_cluster.aliases, mention_count=len(_cluster.mentions))
                            _stats.entities += 1
                            _eids.append(_eid)
                            for _cm in _cluster.mentions:
                                insert_entity_mention(conn, claim_id=_cm.claim_id, entity_id=_eid, mention_text=_cm.mention.text, span_start=_cm.mention.start, span_end=_cm.mention.end)
                                _stats.mentions += 1
                        update_claim_entities(conn, _cid, _eids)
                    conn.commit()
                    insert_llm_call(conn, stage="extract", model=client.model, prompt_version=prompt_version(), latency_ms=int((_time.monotonic()-_started)*1000))
                    for _err in _es.errors:
                        if "LLM call failed" in _err:
                            insert_llm_call(conn, stage="extract", model=client.model, prompt_version=prompt_version(), success=False, error=_err)
                    conn.commit()
                except Exception as _exc:
                    _stats.docs_failed += 1; _stats.errors.append(f"document {_doc['id']}: {_exc!r}")
            stats = _stats
            return {
                "docs_seen": stats.docs_seen,
                "docs_failed": stats.docs_failed,
                "chunks": stats.chunks,
                "llm_calls": stats.llm_calls,
                "claims_llm": stats.claims_llm,
                "claims_inserted": stats.claims_inserted,
                "entities": stats.entities,
                "mentions": stats.mentions,
                "trivial_dropped": stats.trivial_dropped,
                "span_dropped": stats.span_dropped,
                "shape_dropped": stats.shape_dropped,
                "errors": stats.errors,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    @app.post("/api/jobs")
    def api_create_job(req: JobCreate) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            # Cap concurrent jobs (P1 reliability)
            active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if int(active) >= 3:
                raise HTTPException(status_code=429, detail="too many active jobs (max 3)")
            from adonis.db import insert_job

            params = {k: v for k, v in req.model_dump().items() if v is not None}
            jid = insert_job(conn, "pipeline", params)
        finally:
            conn.close()
        t = threading.Thread(target=_run_job_thread, args=(jid, params), daemon=True)
        t.start()
        return {"job_id": jid, "status": "queued", "params": params}

    @app.get("/api/jobs")
    def api_list_jobs(limit: int = Query(20, ge=1, le=100)) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            rows = conn.execute("SELECT id, kind, status, params_json, result_json, error, created_at, started_at, finished_at FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            items = []
            for r in rows:
                items.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "status": r["status"],
                        "params": json.loads(r["params_json"]) if r["params_json"] else {},
                        "result": json.loads(r["result_json"]) if r["result_json"] else None,
                        "error": r["error"],
                        "created_at": r["created_at"],
                        "started_at": r["started_at"],
                        "finished_at": r["finished_at"],
                    }
                )
            return {"items": items}
        finally:
            conn.close()

    @app.get("/api/jobs/{job_id}")
    def api_get_job(job_id: str) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT id, kind, status, params_json, result_json, error, created_at, started_at, finished_at FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="job not found")
            return {
                "id": row["id"],
                "kind": row["kind"],
                "status": row["status"],
                "params": json.loads(row["params_json"]) if row["params_json"] else {},
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "error": row["error"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Labels (pool + staged review)
    # ------------------------------------------------------------------
    @app.get("/api/labels/pending")
    def api_pending_labels(pool: str = Query("entity"), limit: int = Query(20, ge=1, le=50)) -> dict[str, object]:
        from adonis.cli.label_pairs import fetch_pending_pairs

        apply_migrations()
        conn = get_conn()
        try:
            pairs = fetch_pending_pairs(conn, limit=limit, pool=pool)
            items = [
                {
                    "a_id": p.a_id,
                    "b_id": p.b_id,
                    "a_doc": p.a_doc,
                    "b_doc": p.b_doc,
                    "a_text": p.a_text,
                    "b_text": p.b_text,
                    "a_doc_id": p.a_doc_id,
                    "b_doc_id": p.b_doc_id,
                    "a_start": p.a_start,
                    "a_end": p.a_end,
                    "b_start": p.b_start,
                    "b_end": p.b_end,
                }
                for p in pairs
            ]
            return {"pool": pool, "items": items}
        finally:
            conn.close()

    @app.post("/api/labels")
    def api_create_label(payload: LabelCreate) -> dict[str, object]:
        """Create a staged label (pending review), not directly into eval_labels."""
        valid = {"genuine_contradiction", "superseded_by_time", "different_scope", "ambiguous", "not_conflicting", "true_negative_near_dup", "true_negative_unrelated"}
        if payload.label not in valid:
            raise HTTPException(status_code=400, detail=f"label must be one of {sorted(valid)}")
        apply_migrations()
        conn = get_conn()
        try:
            # Verify claims exist and get doc ids/spans
            ra = conn.execute("SELECT id, document_id, citation_span_start, citation_span_end FROM claims WHERE id=?", (payload.claim_a_id,)).fetchone()
            rb = conn.execute("SELECT id, document_id, citation_span_start, citation_span_end FROM claims WHERE id=?", (payload.claim_b_id,)).fetchone()
            if ra is None or rb is None:
                raise HTTPException(status_code=404, detail="claim not found")
            sid = insert_staged_label(
                conn,
                claim_a_id=payload.claim_a_id,
                claim_b_id=payload.claim_b_id,
                doc_a_id=ra["document_id"],
                doc_b_id=rb["document_id"],
                span_a_start=ra["citation_span_start"],
                span_a_end=ra["citation_span_end"],
                span_b_start=rb["citation_span_start"],
                span_b_end=rb["citation_span_end"],
                label=payload.label,
                labeled_by=payload.labeled_by,
                notes=payload.notes,
            )
            row = conn.execute("SELECT * FROM staged_labels WHERE id=?", (sid,)).fetchone()
            return {
                "id": row["id"],
                "status": row["status"],
                "label": row["label"],
                "claim_a_id": row["claim_a_id"],
                "claim_b_id": row["claim_b_id"],
                "created_at": row["created_at"],
            }
        finally:
            conn.close()

    @app.get("/api/labels/staged")
    def api_list_staged(
        status: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            where = ""
            params: list[object] = []
            if status:
                where = " WHERE status=?"
                params.append(status)
            total = conn.execute(f"SELECT COUNT(*) FROM staged_labels{where}", tuple(params)).fetchone()[0]  # type: ignore[arg-type]
            rows = conn.execute(
                f"SELECT id, claim_a_id, claim_b_id, doc_a_id, doc_b_id, label, notes, labeled_by, status, reviewed_by, reviewed_at, created_at FROM staged_labels{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            staged_pending = conn.execute("SELECT COUNT(*) FROM staged_labels WHERE status='pending'").fetchone()[0]
            items = [
                {
                    "id": r["id"],
                    "claim_a_id": r["claim_a_id"],
                    "claim_b_id": r["claim_b_id"],
                    "doc_a_id": r["doc_a_id"],
                    "doc_b_id": r["doc_b_id"],
                    "label": r["label"],
                    "notes": r["notes"],
                    "labeled_by": r["labeled_by"],
                    "status": r["status"],
                    "reviewed_by": r["reviewed_by"],
                    "reviewed_at": r["reviewed_at"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
            return {"items": items, "total": int(total), "staged_pending": int(staged_pending), "limit": limit, "offset": offset}
        finally:
            conn.close()

    @app.post("/api/labels/staged/{staged_id}/review")
    def api_review_staged(staged_id: str, review: StagedReview) -> dict[str, object]:
        if review.action not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="action must be approve|reject")
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT * FROM staged_labels WHERE id=?", (staged_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="staged label not found")
            if row["status"] != "pending":
                raise HTTPException(status_code=409, detail=f"already {row['status']}")
            new_status = "approved" if review.action == "approve" else "rejected"
            now = datetime.now(UTC).isoformat()
            conn.execute(
                "UPDATE staged_labels SET status=?, reviewed_by=?, reviewed_at=?, notes=COALESCE(?, notes) WHERE id=?",
                (new_status, review.reviewed_by, now, review.notes, staged_id),
            )
            # On approve, copy to eval_labels (used_in_eval=1 implicitly via insert)
            if new_status == "approved":
                from adonis.db import insert_eval_label

                insert_eval_label(
                    conn,
                    claim_a_id=row["claim_a_id"],
                    claim_b_id=row["claim_b_id"],
                    doc_a_id=row["doc_a_id"],
                    doc_b_id=row["doc_b_id"],
                    span_a_start=row["span_a_start"],
                    span_a_end=row["span_a_end"],
                    span_b_start=row["span_b_start"],
                    span_b_end=row["span_b_end"],
                    label=row["label"],
                    labeled_by=row["labeled_by"],
                    notes=row["notes"],
                )
            conn.commit()
            updated = conn.execute("SELECT * FROM staged_labels WHERE id=?", (staged_id,)).fetchone()
            return {"id": updated["id"], "status": updated["status"], "reviewed_by": updated["reviewed_by"], "reviewed_at": updated["reviewed_at"]}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Connections + Drive live sync
    # ------------------------------------------------------------------
    @app.get("/api/connections")
    def api_list_connections() -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            try:
                rows = conn.execute("SELECT * FROM connections ORDER BY created_at DESC").fetchall()
            except sqlite3.OperationalError:
                return {"items": []}
            items = []
            for r in rows:
                items.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "name": r["name"],
                        "config": json.loads(r["config_json"]) if r["config_json"] else {},
                        "status": r["status"],
                        "last_sync_at": r["last_sync_at"],
                        "last_sync_stats": json.loads(r["last_sync_stats_json"]) if r["last_sync_stats_json"] else None,
                        "error": r["error"],
                        "created_at": r["created_at"],
                    }
                )
            return {"items": items}
        finally:
            conn.close()

    @app.post("/api/connections")
    def api_create_connection(payload: ConnectionCreate) -> dict[str, object]:
        if payload.kind not in ("local", "notion", "drive"):
            raise HTTPException(status_code=400, detail="kind must be local|notion|drive")
        apply_migrations()
        conn = get_conn()
        try:
            cid = uuid.uuid4().hex
            name = payload.name or payload.kind
            conn.execute(
                "INSERT INTO connections (id, kind, name, config_json, status, created_at) VALUES (?, ?, ?, ?, 'disconnected', ?)",
                (cid, payload.kind, name, json.dumps(payload.config), datetime.now(UTC).isoformat()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM connections WHERE id=?", (cid,)).fetchone()
            return {
                "id": row["id"],
                "kind": row["kind"],
                "name": row["name"],
                "config": json.loads(row["config_json"]),
                "status": row["status"],
                "created_at": row["created_at"],
            }
        finally:
            conn.close()

    @app.delete("/api/connections/{cid}")
    def api_delete_connection(cid: str) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM connections WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="connection not found")
            conn.execute("DELETE FROM connections WHERE id=?", (cid,))
            conn.commit()
            return {"deleted": cid}
        finally:
            conn.close()

    @app.post("/api/connections/{cid}/sync")
    def api_sync_connection(cid: str) -> dict[str, object]:
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT * FROM connections WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="connection not found")
            kind = row["kind"]
            config = json.loads(row["config_json"]) if row["config_json"] else {}
            conn.execute("UPDATE connections SET status='syncing', error=NULL WHERE id=?", (cid,))
            conn.commit()
        finally:
            conn.close()

        # Dispatch sync by kind
        stats: dict[str, object] = {}
        error: str | None = None
        try:
            if kind == "local":
                path = config.get("path")
                if not path:
                    raise ValueError("local connection missing config.path")
                p = Path(path)
                if not p.exists():
                    raise FileNotFoundError(f"path not found: {p}")
                from adonis.ingest.pipeline import ingest_corpus as do_ingest

                c2 = get_conn()
                try:
                    s = do_ingest(p, c2)
                    stats = {"files_seen": s.files_seen, "inserted": s.inserted, "duplicates": s.duplicates, "failed": s.failed, "skipped": s.skipped, "ignored": s.ignored, "errors": s.errors, "notes": s.notes}
                finally:
                    c2.close()
            elif kind == "notion":
                # Expect config.path to a zip file
                path = config.get("path")
                if not path:
                    raise ValueError("notion connection missing config.path (zip file)")
                from adonis.db import insert_document
                from adonis.ingest.notion import parse_notion_export

                records = parse_notion_export(Path(path))
                c2 = get_conn()
                try:
                    inserted = 0
                    for rec in records:
                        if insert_document(c2, rec):
                            inserted += 1
                    stats = {"files_seen": len(records), "inserted": inserted}
                finally:
                    c2.close()
            elif kind == "drive":
                from adonis.ingest.drive_sync import sync_drive

                res = sync_drive(cid)
                stats = {"files_seen": res.files_seen, "inserted": res.inserted, "errors": res.errors}
                if res.errors:
                    error = "; ".join(res.errors[:3])
            else:
                raise ValueError(f"unknown kind {kind}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stats = {"error": error}

        # Update connection row
        c3 = get_conn()
        try:
            now = datetime.now(UTC).isoformat()
            if error:
                c3.execute("UPDATE connections SET status='error', error=?, last_sync_at=?, last_sync_stats_json=? WHERE id=?", (error, now, json.dumps(stats), cid))
            else:
                c3.execute("UPDATE connections SET status='connected', error=NULL, last_sync_at=?, last_sync_stats_json=? WHERE id=?", (now, json.dumps(stats), cid))
            c3.commit()
        finally:
            c3.close()
        return {"connection_id": cid, "kind": kind, "stats": stats, "error": error}

    @app.get("/api/connections/drive/auth")
    def api_drive_auth() -> dict[str, object]:
        from adonis.ingest.drive_sync import auth_url, is_configured

        if not is_configured():
            raise HTTPException(status_code=400, detail="Google OAuth not configured — set ADONIS_GOOGLE_CLIENT_ID and ADONIS_GOOGLE_CLIENT_SECRET in Settings → Connections or .env")
        url = auth_url()
        return {"auth_url": url}

    @app.get("/api/connections/drive/callback")
    def api_drive_callback(code: str | None = Query(None), state: str | None = Query(None), error: str | None = Query(None)) -> HTMLResponse:
        if error:
            return HTMLResponse(f"<h3>Drive auth failed</h3><p>{error}</p><p><a href='/'>back to console</a></p>", status_code=400)
        if not code:
            raise HTTPException(status_code=400, detail="missing code")
        from adonis.ingest.drive_sync import exchange_code

        try:
            tok = exchange_code(code)
        except Exception as exc:
            return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{exc}</pre><p><a href='/'>back</a></p>", status_code=502)
        # Persist refresh token to .env and create a drive connection if missing
        refresh = tok.get("refresh_token", "")
        access = tok.get("access_token", "")
        patch: dict[str, str] = {}
        if refresh:
            patch["ADONIS_GOOGLE_REFRESH_TOKEN"] = refresh
        if access:
            patch["ADONIS_GOOGLE_ACCESS_TOKEN"] = access
        if patch:
            save_settings(patch)
            reload_settings()
        # Ensure a drive connection exists
        apply_migrations()
        conn = get_conn()
        try:
            existing = conn.execute("SELECT id FROM connections WHERE kind='drive' LIMIT 1").fetchone()
            if existing is None:
                cid = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO connections (id, kind, name, config_json, status, created_at) VALUES (?, ?, ?, ?, 'connected', ?)",
                    (cid, "drive", "Google Drive", json.dumps({"via": "oauth", "scope": "drive.readonly"}), datetime.now(UTC).isoformat()),
                )
            else:
                conn.execute("UPDATE connections SET status='connected', error=NULL WHERE kind='drive'")
            conn.commit()
        finally:
            conn.close()
        return HTMLResponse("<h3>Google Drive connected</h3><p>Tokens saved to .env. You can close this tab and return to the console.</p><p><a href='/'>back to console</a></p>")

    @app.post("/api/connections/drive/sync")
    def api_drive_sync() -> dict[str, object]:
        from adonis.ingest.drive_sync import sync_drive

        res = sync_drive()
        # Update connection row if present
        apply_migrations()
        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM connections WHERE kind='drive' LIMIT 1").fetchone()
            if row:
                now = datetime.now(UTC).isoformat()
                stats = {"files_seen": res.files_seen, "inserted": res.inserted, "errors": res.errors}
                if res.errors:
                    conn.execute("UPDATE connections SET status='error', error=?, last_sync_at=?, last_sync_stats_json=? WHERE id=?", ("; ".join(res.errors[:3]), now, json.dumps(stats), row["id"]))
                else:
                    conn.execute("UPDATE connections SET status='connected', last_sync_at=?, last_sync_stats_json=? WHERE id=?", (now, json.dumps(stats), row["id"]))
                conn.commit()
        finally:
            conn.close()
        return {"files_seen": res.files_seen, "inserted": res.inserted, "errors": res.errors}

    return app


app = create_app()
