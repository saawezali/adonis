"""Eval report: metrics over the labeled set.

Loads eval_labels + judge outputs + verification from the store, prints a
per-category P/R/F1 table plus micro/macro averages, detection rates, and
the citation-faithfulness rate; writes the same numbers as JSON to
reports/eval.json.

Run: python scripts/eval_report.py [--json PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adonis.config import get_settings
from adonis.db import apply_migrations, get_conn
from adonis.eval.harness import eval_json, run_eval

_HEADER = (
    f"{'category':<26} {'tp':>3} {'fp':>3} {'fn':>3} "
    f"{'precision':>9} {'recall':>7} {'f1':>7}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval report over the labeled set.")
    ap.add_argument("--json", dest="json_path", default=None, help="artifact path (default reports/eval.json)")
    args = ap.parse_args()

    apply_migrations()
    conn = get_conn()
    try:
        report = run_eval(conn)
    finally:
        conn.close()

    print(f"eval report  ({report.generated_at})")
    print(f"labeled pairs: {report.n_labeled}")
    print(f"judged:        {report.n_judged}   verified: {report.n_verified}")
    print()
    print(_HEADER)
    print("-" * len(_HEADER))
    for c in report.metrics.categories:
        p = f"{c.precision:.3f}" if c.precision is not None else "   n/a"
        r = f"{c.recall:.3f}" if c.recall is not None else "   n/a"
        f1 = f"{c.f1:.3f}" if c.f1 is not None else "   n/a"
        print(f"{c.category:<26} {c.tp:>3} {c.fp:>3} {c.fn:>3} {p:>9} {r:>7} {f1:>7}")
    print("-" * len(_HEADER))
    micro = report.metrics.micro
    macro = report.metrics.macro_f1
    print(
        f"{'micro':<26} {'':>3} {'':>3} {'':>3} {micro['precision']:>9.3f}"
        f" {micro['recall']:>7.3f} {micro['f1']:>7.3f}"
    )
    print(f"macro f1:               {macro if macro is not None else 'n/a'}")
    print(
        f"detection recall:       {report.metrics.detection_recall if report.metrics.detection_recall is not None else 'n/a'}"
    )
    print(
        f"detection precision:    {report.metrics.detection_precision if report.metrics.detection_precision is not None else 'n/a'}"
    )
    print(
        f"citation faithfulness:  {report.metrics.citation_faithfulness if report.metrics.citation_faithfulness is not None else 'n/a'}"
    )

    json_path = args.json_path or str(get_settings().reports_dir / "eval.json")
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(eval_json(report), encoding="utf-8")
    print(f"\nartifact: {path}")


if __name__ == "__main__":
    main()