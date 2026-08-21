"""Eval harness over the store (PLAN.md M5).

Loads every eval_labels row, joins it to the judge output for the same
candidate pair (either ordering), pulls verification results, and produces
a SummaryMetrics with per-category P/R/F1, micro/macro, detection rates,
and citation faithfulness. The report is also available as a plain dict
for the JSON artifact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from adonis.eval.metrics import (
    SummaryMetrics,
    citation_faithfulness,
    detection_rates,
    macro_f1,
    micro_metrics,
    per_category_metrics,
)


@dataclass(frozen=True)
class EvalReport:
    metrics: SummaryMetrics
    n_labeled: int
    n_judged: int
    n_verified: int
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        categories = {}
        for c in self.metrics.categories:
            categories[c.category] = {
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
                "precision": c.precision,
                "recall": c.recall,
                "f1": c.f1,
            }
        return {
            "generated_at": self.generated_at,
            "n_labeled": self.n_labeled,
            "n_judged": self.n_judged,
            "n_verified": self.n_verified,
            "categories": categories,
            "micro": self.metrics.micro,
            "macro_f1": self.metrics.macro_f1,
            "detection_recall": self.metrics.detection_recall,
            "detection_precision": self.metrics.detection_precision,
            "citation_faithfulness": self.metrics.citation_faithfulness,
        }


def run_eval(conn: sqlite3.Connection) -> EvalReport:
    """Compute the eval report from the current store contents."""
    labeled = conn.execute(
        "SELECT el.claim_a_id, el.claim_b_id, el.label FROM eval_labels el"
    ).fetchall()
    predictions: list[tuple[str | None, str | None]] = []
    overall_passes: list[int | bool] = []
    n_judged = 0
    n_verified = 0
    for row in labeled:
        hit = conn.execute(
            "SELECT jo.label, vr.overall_pass"
            " FROM judge_outputs jo"
            " JOIN candidate_pairs cp ON cp.id = jo.candidate_pair_id"
            " LEFT JOIN verification_results vr ON vr.judge_output_id = jo.id"
            " WHERE (cp.claim_a_id = ? AND cp.claim_b_id = ?)"
            "    OR (cp.claim_a_id = ? AND cp.claim_b_id = ?)",
            (row["claim_a_id"], row["claim_b_id"], row["claim_b_id"], row["claim_a_id"]),
        ).fetchone()
        predicted = hit["label"] if hit is not None else None
        predictions.append((row["label"], predicted))
        if hit is not None:
            n_judged += 1
            if hit["overall_pass"] is not None:
                n_verified += 1
                overall_passes.append(hit["overall_pass"])

    categories = per_category_metrics(predictions)
    detection_recall, detection_precision = detection_rates(predictions)
    return EvalReport(
        metrics=SummaryMetrics(
            categories=categories,
            micro=micro_metrics(categories),
            macro_f1=macro_f1(categories),
            detection_recall=detection_recall,
            detection_precision=detection_precision,
            citation_faithfulness=citation_faithfulness(overall_passes),
        ),
        n_labeled=len(labeled),
        n_judged=n_judged,
        n_verified=n_verified,
        generated_at=datetime.now(UTC).isoformat(),
    )


def labeled_pair_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM eval_labels").fetchone()
    return int(row[0])


def eval_json(report: EvalReport) -> str:
    """Serializable JSON artifact for scripts/eval_report.py."""
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)