# Security

## Reporting

Please report vulnerabilities via **GitHub Security Advisories** or Issues (label `security`). Do not disclose publicly until triaged. Expect a response within 3 business days.

## Threat model (single-user local app)

* **Trust boundary:** `data/corpus` is your files; the console binds to `127.0.0.1:8000` (not `0.0.0.0`).
* **Secrets at rest:** `.env` is `chmod 600`; `save_settings` rejects `\n`/`\r` injection and enforces `ADONIS_` prefix.
* **Ingestion sandbox:** `/api/ingest` `path` must resolve inside `ADONIS_CORPUS_DIR` (403 otherwise). Notion zip entries with `/` or `..` are dropped (zip-slip). Corpus `LIKE` queries escape `%`, `_`, `\`.
* **Uploads:** Streamed 1 MB chunks, early 413 at `ADONIS_MAX_UPLOAD_MB` (default 100), max 50 files, per-request cap, temp dir deleted after ingest.
* **OAuth:** Drive `state` is `secrets.token_urlsafe(16)` per request; refresh tokens auto-persist after rotation with `save_settings`.
* **DB:** `-IN` batches (450/900) avoid `too many SQL variables`; `candidate_pairs` ordered `a<b` via Python + SQLite triggers `trg_candidate_order_*`; flags unique per pair; verification must pass (`overall_pass`) before a flag surfaces.

## Hardening checklist for self-hosting

* Bind behind auth proxy if exposing beyond localhost.
* Rotate `ADONIS_LLM_API_KEY` via console Settings → Test → Save.
* Run `pip-audit` / `trivy image adonis:0.1.0` before release (CI does this on push).

## Dependencies

Run `pip-audit` locally: `pip install pip-audit && pip-audit`. Dependabot is enabled for `pyproject.toml`+`Dockerfile`.
