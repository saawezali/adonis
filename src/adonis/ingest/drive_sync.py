"""Google Drive live-sync (single-user, local).

Auth flow: OAuth 2.0 Authorization Code (PKCE not required for web app).
Tokens are stored in .env (ADONIS_GOOGLE_REFRESH_TOKEN etc) and also in
the connections row config_json for the 'drive' kind. A background sync
polls Drive changes and ingests new/updated docs.

For v1 the sync is export-based: it downloads .docx/.md/.txt files via
the Drive API and feeds them through the existing ingest pipeline. .gdoc
(Google Docs native) are exported as docx via the API.

If google-api-python-client is not installed or credentials missing, the
module degrades gracefully — connections stay 'disconnected' and the UI
explains what to configure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adonis.config import get_settings

# ---------------------------------------------------------------------------
# Helpers that never require google libs import
# ---------------------------------------------------------------------------

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def is_configured() -> bool:
    s = get_settings()
    return bool(s.google_client_id and s.google_client_secret)


def auth_url(state: str | None = None) -> str:
    """Build the Google OAuth consent URL for Drive readonly. Random state if not supplied."""
    import secrets as _secrets

    s = get_settings()
    if not s.google_client_id:
        raise RuntimeError("ADONIS_GOOGLE_CLIENT_ID not set — add it in Settings → Connections")
    redirect = s.google_redirect_uri or "http://127.0.0.1:8000/api/connections/drive/callback"
    from urllib.parse import urlencode

    if state is None:
        state = _secrets.token_urlsafe(16)
    params = {
        "client_id": s.google_client_id,
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": " ".join(DRIVE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def exchange_code(code: str) -> dict[str, Any]:
    """Exchange authorization code for tokens. Returns token response dict."""
    import httpx

    s = get_settings()
    redirect = s.google_redirect_uri or "http://127.0.0.1:8000/api/connections/drive/callback"
    data = {
        "code": code,
        "client_id": s.google_client_id or "",
        "client_secret": s.google_client_secret or "",
        "redirect_uri": redirect,
        "grant_type": "authorization_code",
    }
    resp = httpx.post("https://oauth2.googleapis.com/token", data=data, timeout=20)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


@dataclass
class SyncResult:
    files_seen: int = 0
    inserted: int = 0
    changed: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def sync_drive(conn_row_id: str | None = None, *, max_files: int = 100) -> SyncResult:
    """One polling sync: list Drive files, download new ones, ingest.

    If no connection row supplied, uses env tokens. Returns stats; does not
    raise on Drive errors (they are recorded in errors).
    """
    result = SyncResult()
    try:
        return _sync_inner(max_files=max_files)
    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
        return result


def _sync_inner(*, max_files: int = 100) -> SyncResult:
    import httpx

    from adonis.db import get_conn, insert_document
    from adonis.ingest.base import DocumentRecord
    from adonis.normalize.text import normalize_text

    s = get_settings()
    # Tokens from .env or env var
    refresh = os.getenv("ADONIS_GOOGLE_REFRESH_TOKEN", "")
    access = os.getenv("ADONIS_GOOGLE_ACCESS_TOKEN", "")
    if not refresh and not access:
        return SyncResult(errors=["not authenticated — connect Drive in Connections tab"])

    # Refresh access token if needed; persist new token
    if refresh and not access:
        tok = _refresh_access_token(refresh)
        access = tok.get("access_token", "")
        if access and tok.get("refresh_token"):
            # Some flows return rotated refresh_token; persist
            try:
                from adonis.config import save_settings

                save_settings(
                    {
                        "ADONIS_GOOGLE_ACCESS_TOKEN": access,
                        "ADONIS_GOOGLE_REFRESH_TOKEN": tok.get("refresh_token", refresh),
                    }
                )
            except Exception:
                pass

    if not access:
        return SyncResult(errors=["no access token — re-authenticate"])

    headers = {"Authorization": f"Bearer {access}"}
    # List files: only exportable docs + uploads
    # We filter mimeTypes to docs we can handle.
    mime_q = "(mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mimeType='text/plain' or mimeType='text/markdown')"
    params = {"q": f"trashed=false and ({mime_q})", "pageSize": max_files, "fields": "files(id,name,mimeType,modifiedTime)"}
    resp = httpx.get("https://www.googleapis.com/drive/v3/files", headers=headers, params=params, timeout=20)
    if resp.status_code == 401 and refresh:
        tok = _refresh_access_token(refresh)
        access = tok.get("access_token", "")
        if access:
            try:
                from adonis.config import save_settings

                save_settings({"ADONIS_GOOGLE_ACCESS_TOKEN": access})
            except Exception:
                pass
        headers = {"Authorization": f"Bearer {access}"}
        resp = httpx.get("https://www.googleapis.com/drive/v3/files", headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    result = SyncResult(files_seen=len(files))
    conn = get_conn()
    try:
        for f in files:
            fid = f["id"]
            name = f.get("name", fid)
            mime = f.get("mimeType", "")
            try:
                text = _download_file(fid, mime, headers)
                if not text.strip():
                    continue
                rec = DocumentRecord(
                    source="gdrive",
                    source_id=fid,
                    title=Path(name).stem,
                    path=f"gdrive://{fid}/{name}",
                    format="docx" if "document" in mime else "txt",
                    raw_text=normalize_text(text),
                    metadata={"drive_id": fid, "drive_name": name, "mimeType": mime},
                )
                inserted = insert_document(conn, rec)
                if inserted:
                    result.inserted += 1
                else:
                    result.changed += 0  # deduped
            except Exception as exc:
                result.errors.append(f"{name}: {exc!r}")
    finally:
        conn.close()
    return result


def _refresh_access_token(refresh_token: str) -> dict[str, Any]:
    import httpx

    s = get_settings()
    data = {
        "client_id": s.google_client_id or "",
        "client_secret": s.google_client_secret or "",
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    resp = httpx.post("https://oauth2.googleapis.com/token", data=data, timeout=20)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _download_file(file_id: str, mime: str, headers: dict[str, str]) -> str:
    import httpx

    if mime == "application/vnd.google-apps.document":
        # Export Google Doc as docx then parse — simpler: export as text/plain
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
        resp = httpx.get(url, headers=headers, params={"mimeType": "text/plain"}, timeout=30)
    else:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        resp = httpx.get(url, headers=headers, params={"alt": "media"}, timeout=30)
    resp.raise_for_status()
    # Try utf-8 decode; fallback to replace
    try:
        return resp.text
    except Exception:
        return resp.content.decode("utf-8", errors="replace")
