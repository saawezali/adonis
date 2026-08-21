"""Generate synthetic MIT-licensed sample corpus (5 docs, contradictions).

Run: python scripts/generate_sample_corpus.py [--out data/corpus/sample]

Creates 5 small markdown files exercising the 4 taxonomy labels:
 - genuine_contradiction: Atlas ship month March vs July (same scope/time window, different value)
 - superseded_by_time: pricing €500 (Mar) → €600 (Jul update)
 - different_scope: onboarding required for EU vs US (scope mismatch, not a bug)
 - ambiguous: hiring freeze ??? (underspecified)
 - not_conflicting restatement duplicate (Atlas is our flagship) for TN/duplicate check

All content is synthetic, MIT — you own it — no 3rd-party data bundled.
Expected demo pipeline: ≥1 flag (demo judge is coarse, may flag 2-3).

See README Quickstart.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FILES: dict[str, str] = {
    "01_roadmap.md": """# Roadmap — Project Atlas

Atlas ships in March.

Atlas is our flagship product.

The launch budget is 500 euros.

Onboarding is required for EU customers.
""",
    "02_finance.md": """# Finance — Atlas Budget

Atlas ships in July.

The launch budget is 600 euros.

Onboarding is required for US customers.
""",
    "03_product.md": """# Product — Atlas Overview

Atlas is our flagship product.

We are considering a hiring freeze for Q3.
""",
    "04_ops_eu.md": """# Ops — EU Rollout

Onboarding is required for EU customers.

Support window is 9am–5pm CET.
""",
    "05_ops_us.md": """# Ops — US Rollout

Onboarding is required for US customers.

Support window is 9am–5pm EST.
""",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic sample corpus (MIT).")
    ap.add_argument("--out", default="data/corpus/sample", help="output dir (default data/corpus/sample)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for name, text in FILES.items():
        p = out / name
        if p.exists() and not args.force:
            print(f"skip {p} (exists, use --force)")
            continue
        p.write_text(text, encoding="utf-8")
        written += 1
        print(f"wrote {p}")

    print(f"\ndone — {written} files in {out}")
    print("Try: python scripts/ingest_corpus.py --in", out)
    print("     python scripts/extract_claims.py && python scripts/run_pipeline.py && python scripts/render_report.py")


if __name__ == "__main__":
    main()
