"""M4 pipeline CLI: claims -> candidates -> judge -> verify -> flags.

Thin wrapper over adonis.pipeline; --llm uses the configured judge-tier
provider (see .env / the web console), otherwise a deterministic demo
judge + verifier exercise the flow offline.
"""

from __future__ import annotations

import argparse

from adonis.db import apply_migrations, get_conn
from adonis.judge.demo import DemoJudge
from adonis.llm.client import get_client
from adonis.pair.embed import load_embedder
from adonis.pipeline import PipelineStats, run, wipe_pipeline_rows
from adonis.verify.demo import DemoVerifier


def _report(stats: PipelineStats) -> None:
    print("=" * 76)
    print(f"claims:            {stats.claims}")
    print(f"candidates:        {stats.candidates} ({stats.embedding_pairs} embedding, {stats.entity_pairs} entity)")
    print(f"new candidates:    {stats.new_candidates}")
    print(f"selected for judge: {stats.selected_for_judge}")
    print(f"pairs judged:      {stats.pairs_judged}")
    print(f"judge failures:    {stats.judge_failures}")
    print(f"verified:          {stats.verified}")
    print(f"verification fails: {stats.verification_failures}")
    faithfulness = stats.citation_faithfulness
    if faithfulness is not None:
        print(f"citation faithfulness: {faithfulness:.0%}")
    print(f"flags:             {stats.flags}")
    print("render report:     python scripts/render_report.py")
    for err in stats.errors:
        print(f"  error: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 pipeline: candidates + judge + verify + flags.")
    ap.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="use the real judge/verifier LLMs (requires ADONIS_LLM_API_KEY); default demo",
    )
    ap.add_argument("--max-pairs", type=int, default=None, help="cap judged pairs")
    ap.add_argument("--top-k", type=int, default=None, help="embedding neighbors per claim")
    ap.add_argument("--selected-per-claim", type=int, default=None, help="pairs selected for judge")
    ap.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="wipe candidates/judge/verify/flags before running",
    )
    args = ap.parse_args()

    apply_migrations()
    conn = get_conn()
    try:
        if args.refresh:
            wipe_pipeline_rows(conn)
        if args.llm:
            client = get_client("judge")
            verifier = get_client("judge")
        else:
            client = DemoJudge()
            verifier = DemoVerifier()
        embedder = load_embedder()
        stats = run(
            conn,
            client,
            embedder,
            verifier,
            top_k=args.top_k,
            selected_per_claim=args.selected_per_claim,
            max_pairs=args.max_pairs,
        )
    finally:
        conn.close()

    _report(stats)


if __name__ == "__main__":
    main()