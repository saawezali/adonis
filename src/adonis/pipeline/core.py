"""M4 pipeline core: claims -> candidates -> judge -> verify -> flags.

Runs over the whole store:
  1. load claims, embed them (sentence-transformers; local, no API key),
  2. build candidate pairs (top-K + entity overlap, materialized in
     candidate_pairs),
  3. judge the selected pairs with the judge-tier LLM (or a deterministic
     demo judge),
  4. verify every judge output: lexical span match (verbatim + fuzzy) and
     LLM entailment for both cited spans; verification_results rows are
     written for every judge output,
  5. surface a flag ONLY when verification passes (overall_pass = 1);
     flags.final_confidence comes from the calibrated model.

Client/verifier/embedder are injected; the CLI and web console pick demo
vs real instances.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from adonis.db import (
    insert_candidate_pair,
    insert_flag,
    insert_judge_output,
    insert_llm_call,
    insert_verification_result,
    load_claims,
    load_unjudged_pairs,
)
from adonis.judge.classify import (
    ClaimView,
    JudgeResult,
    judge_pair,
)
from adonis.judge.classify import (
    prompt_version as judge_prompt_version,
)
from adonis.llm.client import LLMClient
from adonis.pair.candidates import build_candidate_rows, claim_rows_from_db
from adonis.pair.embed import Embedder
from adonis.score.confidence import build_calibrated_model
from adonis.verify.entailment import prompt_version as entail_prompt_version
from adonis.verify.entailment import verify_entailment
from adonis.verify.span_match import span_match, span_pass

_FLAG_LABELS = {"genuine_contradiction", "superseded_by_time", "different_scope", "ambiguous"}


@dataclass
class PipelineStats:
    claims: int = 0
    candidates: int = 0
    embedding_pairs: int = 0
    entity_pairs: int = 0
    new_candidates: int = 0
    selected_for_judge: int = 0
    pairs_judged: int = 0
    judge_failures: int = 0
    verified: int = 0
    verification_failures: int = 0
    flags: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def citation_faithfulness(self) -> float | None:
        """Fraction of flag-label judge outputs whose verification passed."""
        if self.verified == 0:
            return None
        return (self.verified - self.verification_failures) / self.verified


@dataclass(frozen=True)
class _SideVerification:
    verbatim: bool
    fuzzy: float
    entailment: float
    passed: bool


def _claim_view(row: sqlite3.Row, side: str) -> ClaimView:
    """Build a ClaimView from a judge-batch row ('a' or 'b' side)."""
    prefix = "a" if side == "a" else "b"
    return ClaimView(
        id=row[f"claim_{prefix}_id"],
        text=row[f"text_{prefix}"],
        temporal=_load_dict(row[f"temporal_{prefix}"]),
        scope=_load_dict(row[f"scope_{prefix}"]),
    )


def _span_text(row: sqlite3.Row, side: str, result: JudgeResult) -> str:
    prefix = "a" if side == "a" else "b"
    span_start = row[f"span_{prefix}_start"] + getattr(result, f"span_{prefix}_start")
    span_end = row[f"span_{prefix}_start"] + getattr(result, f"span_{prefix}_end")
    raw = str(row[f"raw_{prefix}"])
    return raw[span_start:span_end]


def _verify_side(
    conn: sqlite3.Connection,
    verifier: LLMClient,
    claim_text: str,
    cited_text: str,
) -> _SideVerification:
    lex = span_match(claim_text, cited_text)
    started = time.monotonic()
    entail, error = verify_entailment(verifier, cited_text, claim_text)
    if error is not None or entail is None:
        insert_llm_call(
            conn,
            stage="verify",
            model=verifier.model,
            prompt_version=entail_prompt_version(),
            success=False,
            error=error,
        )
        return _SideVerification(
            verbatim=lex.verbatim,
            fuzzy=lex.fuzzy_ratio,
            entailment=0.0,
            passed=False,
        )
    insert_llm_call(
        conn,
        stage="verify",
        model=verifier.model,
        prompt_version=entail_prompt_version(),
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    passed = span_pass(lex) and entail.pass_
    return _SideVerification(
        verbatim=lex.verbatim,
        fuzzy=lex.fuzzy_ratio,
        entailment=entail.score,
        passed=passed,
    )


def run(
    conn: sqlite3.Connection,
    client: LLMClient,
    embedder: Embedder,
    verifier: LLMClient,
    *,
    top_k: int | None = None,
    selected_per_claim: int | None = None,
    max_pairs: int | None = None,
) -> PipelineStats:
    """Materialize candidates, judge selected pairs, verify, and flag."""
    stats = PipelineStats()
    calibrated = build_calibrated_model(conn)

    claim_rows = load_claims(conn)
    stats.claims = len(claim_rows)
    if not claim_rows:
        stats.errors.append("no claims in store; run scripts/extract_claims.py first")
        return stats

    candidates, candidate_stats = build_candidate_rows(
        claim_rows_from_db(claim_rows),
        embedder,
        top_k=top_k,
        selected_per_claim=selected_per_claim,
    )
    stats.candidates = candidate_stats.candidates
    stats.embedding_pairs = candidate_stats.embedding_pairs
    stats.entity_pairs = candidate_stats.entity_pairs
    for pair in candidates:
        if insert_candidate_pair(
            conn,
            claim_a_id=pair.claim_a_id,
            claim_b_id=pair.claim_b_id,
            similarity_score=pair.similarity_score,
            entity_overlap=pair.entity_overlap,
            combined_score=pair.combined_score,
            strategy=pair.strategy,
            selected_for_judge=pair.selected_for_judge,
        ):
            stats.new_candidates += 1
    conn.commit()

    pairs = load_unjudged_pairs(conn, limit=max_pairs)
    stats.selected_for_judge = len(pairs)
    for row in pairs:
        started = time.monotonic()
        claim_a = _claim_view(row, "a")
        claim_b = _claim_view(row, "b")
        result, error = judge_pair(client, claim_a, claim_b)
        if error is not None:
            stats.judge_failures += 1
            stats.errors.append(f"pair {row['pair_id']}: {error}")
            insert_llm_call(
                conn,
                stage="judge",
                model=client.model,
                prompt_version=judge_prompt_version(),
                success=False,
                error=error,
            )
            conn.commit()
            continue
        assert result is not None
        judge_output_id = insert_judge_output(
            conn,
            candidate_pair_id=row["pair_id"],
            label=result.label,
            judge_confidence=result.confidence,
            reasoning_text=result.reasoning,
            cited_span_a_start=row["span_a_start"] + result.span_a_start,
            cited_span_a_end=row["span_a_start"] + result.span_a_end,
            cited_span_b_start=row["span_b_start"] + result.span_b_start,
            cited_span_b_end=row["span_b_start"] + result.span_b_end,
            judge_model=client.model,
            prompt_version=judge_prompt_version(),
        )
        stats.pairs_judged += 1
        if result.label in _FLAG_LABELS:
            side_a = _verify_side(conn, verifier, claim_a.text, _span_text(row, "a", result))
            side_b = _verify_side(conn, verifier, claim_b.text, _span_text(row, "b", result))
            overall_pass = side_a.passed and side_b.passed
            insert_verification_result(
                conn,
                judge_output_id=judge_output_id,
                span_a_verbatim=side_a.verbatim,
                span_a_fuzzy=side_a.fuzzy,
                span_a_entailment=side_a.entailment,
                span_a_pass=side_a.passed,
                span_b_verbatim=side_b.verbatim,
                span_b_fuzzy=side_b.fuzzy,
                span_b_entailment=side_b.entailment,
                span_b_pass=side_b.passed,
            )
            stats.verified += 1
            if not overall_pass:
                stats.verification_failures += 1
                stats.errors.append(
                    f"pair {row['pair_id']}: verification failed "
                    f"(a:{side_a.passed} b:{side_b.passed})"
                )
            else:
                final_confidence = calibrated.transform(result.confidence, row["combined_score"])
                insert_flag(
                    conn,
                    candidate_pair_id=row["pair_id"],
                    final_label=result.label,
                    final_confidence=final_confidence,
                )
                stats.flags += 1
                _print_flag(row, result, conn)
        insert_llm_call(
            conn,
            stage="judge",
            model=client.model,
            prompt_version=judge_prompt_version(),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        conn.commit()
    return stats


def _load_dict(raw: object) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _print_flag(row: sqlite3.Row, result: JudgeResult, conn: sqlite3.Connection) -> None:
    span_a_text = row["raw_a"][
        row["span_a_start"] + result.span_a_start : row["span_a_start"] + result.span_a_end
    ]
    span_b_text = row["raw_b"][
        row["span_b_start"] + result.span_b_start : row["span_b_start"] + result.span_b_end
    ]
    print("=" * 76)
    print(f"FLAG [{row['doc_a_title']}]  {row['text_a'].strip()[:90]!r}")
    print(f"   cited: {span_a_text.strip()[:90]!r}")
    print(f"FLAG [{row['doc_b_title']}]  {row['text_b'].strip()[:90]!r}")
    print(f"   cited: {span_b_text.strip()[:90]!r}")
    print(
        f"{result.label}  confidence={result.confidence:.2f}  "
        f"combined={row['combined_score']:.3f}"
    )
    print(f"   reasoning: {result.reasoning.strip()[:200]}")


def wipe_pipeline_rows(conn: sqlite3.Connection) -> None:
    """Drop candidate/judge/verify/flag rows so a run is fully repeatable."""
    for table in ("flags", "verification_results", "judge_outputs", "candidate_pairs"):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()