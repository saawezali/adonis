"""Run the judge over the curated trick set (PLAN.md M3).

Each pair in data/eval/trick_set.jsonl states the label the §1.4 decision
rules require. Without --llm a deterministic DemoJudge is used (exercises
the plumbing, not the model); with --llm the real judge runs (needs the
API key). Exits 1 when any pair is judged differently than expected.

Run: python scripts/eval_trick_set.py [--llm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adonis.judge.classify import ClaimView, judge_pair, prompt_version
from adonis.judge.demo import DemoJudge
from adonis.llm.client import get_client

_TRICK_SET = Path(__file__).parent.parent / "data" / "eval" / "trick_set.jsonl"


def _view(raw: dict[str, object]) -> ClaimView:
    temporal = raw.get("temporal")
    scope = raw.get("scope")
    return ClaimView(
        id="trick",
        text=str(raw["text"]),
        temporal=temporal if isinstance(temporal, dict) else None,
        scope=scope if isinstance(scope, dict) else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the trick set.")
    ap.add_argument(
        "--llm", action="store_true", default=False,
        help="use the configured judge LLM; default: deterministic DemoJudge",
    )
    args = ap.parse_args()

    client = get_client("judge") if args.llm else DemoJudge()
    cases = [
        json.loads(line) for line in _TRICK_SET.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"trick set: {len(cases)} pairs, judge={client.model}, prompt={prompt_version()}")

    mismatches = 0
    for case in cases:
        expected = str(case["expected_label"])
        result, error = judge_pair(
            client,
            _view(case["claim_a"]),
            _view(case["claim_b"]),
        )
        got = "error" if error is not None else (result.label if result is not None else "invalid")
        mark = "OK " if got == expected else "FAIL"
        if got != expected:
            mismatches += 1
        print(f"  [{mark}] expected={expected:<22} got={got:<22} {case['note']}")
        if error is not None:
            print(f"         error: {error}")

    print(f"passed {len(cases) - mismatches}/{len(cases)}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()