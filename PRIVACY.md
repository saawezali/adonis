# Privacy

Adonis is **local-only**.

* **No telemetry, no cloud sync.** Documents, claims, embeddings, flags, and eval labels live in `data/db/adonis.sqlite` on your machine. Reports live in `reports/`.
* **LLM calls:** Only when you run with `--llm` (or Real mode in the console). Then the selected provider (`anthropic` / `openai` / `custom`) receives the chunk or claim pair you are judging. See that provider's privacy policy. Demo mode makes no network calls.
* **Secrets:** `ADONIS_LLM_API_KEY`, `ADONIS_GOOGLE_CLIENT_SECRET`, and Drive `refresh_token` are stored in `.env` with `chmod 600`. They are never logged; the console masks them (`…ABCD`). `ADONIS_GOOGLE_*` uses OAuth `drive.readonly` — read-only, single-user.
* **Logs:** `llm_calls` stores model, prompt_version, latency, and error — not raw document text. Clear it with `DELETE FROM llm_calls;` or `make clean` (also removes DB).
* **Uploads:** Console uploads are streamed to a temp dir, ingested, then deleted. No copy is retained outside the DB.
* **Deletion:** Deleting a document in the console removes its DB row only; the original file on disk is untouched (`api/documents/{id}` DELETE cascades claims/pairs/flags).

If you need stricter isolation, use `custom` provider pointed at a local inference server (Ollama/vLLM) — then no data leaves the machine at all.
