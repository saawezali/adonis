"""Static HTML report over flags (PLAN.md M4).

Renders reports/index.html from the flags table plus context (pairs,
claims, documents, verification), sorted by final_confidence desc so the
most likely contradictions are on top. Source links point at the original
document paths (file:// URLs are opened by most browsers).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import pathname2url

from jinja2 import Environment, FileSystemLoader

from adonis.config import get_settings, provider_for_tier
from adonis.db import get_conn, load_flags_with_context, load_judged_pairs_with_verification

_TEMPLATES_DIR = Path(__file__).parent


def _snippet(text: str, start: int, end: int, width: int = 160) -> str:
    """Add ellipses around the cited span for human-scannable inline quotes."""
    left = max(0, start - width // 2)
    right = min(len(text), end + width // 2)
    prefix = "\u2026" if left > 0 else ""
    suffix = "\u2026" if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def render_report(conn: sqlite3.Connection, out_path: Path) -> Path:
    """Write the HTML report; returns its path."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("template.html.j2")

    flags = []
    for row in load_flags_with_context(conn):
        flags.append(
            {
                "final_label": row["final_label"],
                "final_confidence": float(row["final_confidence"]),
                "text_a": row["text_a"].strip(),
                "text_b": row["text_b"].strip(),
                "doc_a_title": row["doc_a_title"],
                "doc_b_title": row["doc_b_title"],
                "doc_a_path": _file_url(row["doc_a_path"], row["doc_a_title"]),
                "doc_b_path": _file_url(row["doc_b_path"], row["doc_b_title"]),
                "span_a": _snippet(row["raw_a"], row["cited_span_a_start"] or 0, row["cited_span_a_end"] or 0),
                "span_b": _snippet(row["raw_b"], row["cited_span_b_start"] or 0, row["cited_span_b_end"] or 0),
                "span_a_verbatim": bool(row["span_a_verbatim"]),
                "span_b_verbatim": bool(row["span_b_verbatim"]),
                "span_a_fuzzy": float(row["span_a_fuzzy"]),
                "span_b_fuzzy": float(row["span_b_fuzzy"]),
                "span_a_entailment": float(row["span_a_entailment"]),
                "span_b_entailment": float(row["span_b_entailment"]),
                "judge_model": row["judge_model"],
                "judge_prompt_version": row["prompt_version"],
                "reasoning": row["reasoning_text"],
            }
        )

    judged = load_judged_pairs_with_verification(conn)
    checked = [r for r in judged if r["overall_pass"] is not None]
    faithfulness = (
        sum(1 for r in checked if r["overall_pass"] == 1) / len(checked) if checked else None
    )

    html = template.render(
        flags=flags,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        judge_model=flags[0]["judge_model"] if flags else "n/a",
        judge_provider=provider_for_tier(get_settings(), "judge"),
        judge_prompt_version=flags[0]["judge_prompt_version"] if flags else "n/a",
        faithfulness=faithfulness,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _file_url(path: str | None, title: str) -> str:
    if not path:
        return "#"
    # file:// link to the original doc; fall back to the title when missing.
    try:
        return f"file://{pathname2url(str(Path(path).resolve()))}"
    except (OSError, ValueError):
        return f"#doc-{title}"


def main() -> None:
    settings = get_settings()
    out_path = settings.reports_dir / "index.html"
    conn = get_conn()
    try:
        render_report(conn, out_path)
    finally:
        conn.close()
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()