"""Render the HTML flags report (PLAN.md M4).

Reads flags + verification from the store and writes reports/index.html.
See src/adonis/report/render.py for details.

Run: python scripts/render_report.py
"""

from __future__ import annotations

from adonis.config import get_settings
from adonis.db import apply_migrations, get_conn
from adonis.report.render import render_report


def main() -> None:
    apply_migrations()
    conn = get_conn()
    try:
        out = render_report(conn, get_settings().reports_dir / "index.html")
    finally:
        conn.close()
    print(f"report written: {out}")


if __name__ == "__main__":
    main()