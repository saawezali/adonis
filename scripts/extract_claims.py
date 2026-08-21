"""Extract claims + entities from the document store (M2 entry point).

Reads documents (all, or --document-id / --limit), extracts atomic claims
per chunk via the extractor-tier LLM, runs zero-shot NER over each claim's
citation span, canonicalizes mentions into entities, and persists claims,
entities, entity_mentions, and llm_calls rows. One bad document is counted
and skipped; the run continues.

Without `--llm` (no API key needed) a deterministic canned client is used
that emits one claim per chunk, so the whole pipeline can be exercised and
inspected offline. Real extraction requires `--llm` + ADONIS_LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_claim,
    insert_entity_mention,
    insert_llm_call,
    iter_documents,
    update_claim_entities,
    upsert_entity,
)
from adonis.extract.canonicalize import ClaimMention, cluster_mentions
from adonis.extract.claims import (
    ClaimRecord,
    extract_document_claims,
    prompt_version,
    span_text,
)
from adonis.extract.entities import extract_mentions
from adonis.llm.client import LLMClient, get_client

_SENTENCE_RE = re.compile(r"^([^.!?]*[.!?]|\S[^\n]*)", re.DOTALL)
_TEXT_RE = re.compile(r"<text>\n(.*)\n</text>", re.DOTALL)


@dataclass
class ExtractRunStats:
    """Aggregate outcome counts for one extraction run."""

    docs_seen: int = 0
    docs_failed: int = 0
    chunks: int = 0
    llm_calls: int = 0
    claims_llm: int = 0
    claims_inserted: int = 0
    entities: int = 0
    mentions: int = 0
    trivial_dropped: int = 0
    span_dropped: int = 0
    shape_dropped: int = 0
    errors: list[str] = field(default_factory=list)


def _insert_claims_and_entities(
    conn: sqlite3.Connection,
    doc_id: str,
    raw_text: str,
    claims: list[ClaimRecord],
    extraction_model: str,
    extraction_at: str,
    stats: ExtractRunStats,
) -> None:
    """Persist claims; then NER + canonicalization + mentions per claim."""
    for claim in claims:
        claim_id = insert_claim(
            conn,
            document_id=doc_id,
            claim_text=claim.claim_text,
            span_start=claim.span_start,
            span_end=claim.span_end,
            topics=claim.topics,
            temporal=claim.temporal,
            scope=claim.scope,
            triviality_score=claim.triviality_score,
            extraction_model=extraction_model,
            extraction_at=extraction_at,
        )
        stats.claims_inserted += 1
        span_text_value = span_text(raw_text, claim)
        try:
            mentions = extract_mentions(span_text_value, offset=claim.span_start)
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(
                f"document {doc_id}: NER failed for claim {claim_id!r}: {exc!r}"
            )
            continue
        claim_mentions = [ClaimMention(claim_id=claim_id, mention=m) for m in mentions]
        clusters = cluster_mentions(claim_mentions)
        entity_ids: list[str] = []
        for cluster in clusters:
            entity_id = upsert_entity(
                conn,
                canonical_name=cluster.canonical_name,
                aliases=cluster.aliases,
                mention_count=len(cluster.mentions),
            )
            stats.entities += 1
            entity_ids.append(entity_id)
            for cm in cluster.mentions:
                insert_entity_mention(
                    conn,
                    claim_id=cm.claim_id,
                    entity_id=entity_id,
                    mention_text=cm.mention.text,
                    span_start=cm.mention.start,
                    span_end=cm.mention.end,
                )
                stats.mentions += 1
        update_claim_entities(conn, claim_id, entity_ids)
    conn.commit()


def run(
    conn: sqlite3.Connection,
    client: LLMClient | None = None,
    *,
    document_id: str | None = None,
    limit: int | None = None,
) -> ExtractRunStats:
    """Extract claims + entities from the store. Client is injectable for tests."""
    client = client if client is not None else get_client("extractor")
    stats = ExtractRunStats()
    extraction_at = datetime.now(UTC).isoformat()
    for doc in iter_documents(conn, document_id=document_id, limit=limit):
        stats.docs_seen += 1
        started = time.monotonic()
        try:
            claims, extraction_stats = extract_document_claims(client, doc["raw_text"])
            stats.chunks += extraction_stats.chunks
            stats.llm_calls += extraction_stats.llm_calls
            stats.claims_llm += extraction_stats.claims_from_llm
            stats.trivial_dropped += extraction_stats.trivial_dropped
            stats.span_dropped += extraction_stats.span_dropped
            stats.shape_dropped += extraction_stats.shape_dropped
            stats.errors.extend(extraction_stats.errors)
            _insert_claims_and_entities(
                conn,
                doc["id"],
                doc["raw_text"],
                claims,
                client.model,
                extraction_at,
                stats,
            )
            insert_llm_call(
                conn,
                stage="extract",
                model=client.model,
                prompt_version=prompt_version(),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            for err in extraction_stats.errors:
                if "LLM call failed" in err:
                    insert_llm_call(
                        conn,
                        stage="extract",
                        model=client.model,
                        prompt_version=prompt_version(),
                        success=False,
                        error=err,
                    )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            stats.docs_failed += 1
            stats.errors.append(f"document {doc['id']}: {exc!r}")
    return stats


class DemoClient:
    """Deterministic offline stand-in for the extractor tier.

    Emits one claim per chunk (its first sentence) so the pipeline is fully
    exercisable without an API key. The claim span is real and valid.
    """

    model = "demo-canned"

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
        match = _TEXT_RE.search(user)
        if match is None:
            return {"claims": []}
        chunk_text = match.group(1)
        sent = _SENTENCE_RE.match(chunk_text)
        if sent is None:
            return {"claims": []}
        claim_text = sent.group(1).strip()
        if not claim_text or claim_text.endswith(("?", "!")):
            return {"claims": []}
        start = chunk_text.find(claim_text)
        if start < 0:
            return {"claims": []}
        return {
            "claims": [
                {
                    "claim_text": claim_text,
                    "span_start": start,
                    "span_end": start + len(claim_text),
                    "triviality_score": 0.1,
                    "topics": ["demo"],
                    "temporal": None,
                    "scope": None,
                }
            ]
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract claims + entities from the store.")
    ap.add_argument("--document-id", default=None, help="only extract this document")
    ap.add_argument("--limit", type=int, default=None, help="only first N documents")
    ap.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="use the real LLM (requires ADONIS_LLM_API_KEY); default is demo mode",
    )
    args = ap.parse_args()

    apply_migrations()
    conn = get_conn()
    try:
        if args.llm:
            stats = run(conn, document_id=args.document_id, limit=args.limit)
        else:
            stats = run(
                conn,
                client=DemoClient(),
                document_id=args.document_id,
                limit=args.limit,
            )
    finally:
        conn.close()

    print(f"documents seen:       {stats.docs_seen}")
    print(f"documents failed:     {stats.docs_failed}")
    print(f"chunks:               {stats.chunks}")
    print(f"llm calls:            {stats.llm_calls}")
    print(f"claims (from LLM):    {stats.claims_llm}")
    print(f"claims inserted:      {stats.claims_inserted}")
    print(f"entities:             {stats.entities}")
    print(f"mentions:             {stats.mentions}")
    print(f"trivial dropped:      {stats.trivial_dropped}")
    print(f"invalid-span dropped: {stats.span_dropped}")
    print(f"shape dropped:        {stats.shape_dropped}")
    for err in stats.errors:
        print(f"  error: {err}")


if __name__ == "__main__":
    main()
