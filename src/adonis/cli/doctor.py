"""adonis-doctor — pre-flight checks for first-run ease.

Run: python -m adonis.cli.doctor  or  adonis-doctor
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from adonis import __version__
    from adonis.config import get_settings
    from adonis.db import apply_migrations

    print(f"Adonis {__version__} — doctor")
    ok = True

    # Python
    print(f"  python: {sys.version.split()[0]} (need >=3.11) —", end=" ")
    if sys.version_info >= (3, 11):
        print("ok")
    else:
        print("FAIL")
        ok = False

    # Settings
    try:
        s = get_settings()
        print(f"  db_path: {s.db_path} — exists? {s.db_path.parent.exists()} — ok")
        print(f"  corpus_dir: {s.corpus_dir} — exists? {s.corpus_dir.exists()} — {'ok' if s.corpus_dir.exists() else 'will be created'}")
        print(f"  embedding_model: {s.embedding_model}")
        print(f"  extractor: {s.extractor_model} ({s.llm_provider})")
        print(f"  judge: {s.judge_model}")
    except Exception as exc:
        print(f"  config FAIL: {exc!r}")
        ok = False

    # Migrations
    try:
        apply_migrations()
        print("  migrations: ok")
    except Exception as exc:
        print(f"  migrations FAIL: {exc!r}")
        ok = False

    # Prompts
    prompts = Path(__file__).parent.parent / "llm" / "prompts"
    for name in ("claims_v1.txt", "judge_v1.txt", "entail_v1.txt"):
        p = prompts / name
        print(f"  prompt {name}: {'ok' if p.exists() else 'MISSING'}")
        if not p.exists():
            ok = False

    # Disk
    import shutil

    free = shutil.disk_usage(".").free / (1024**3)
    print(f"  disk free: {free:.1f} GB")
    if free < 1:
        print("    WARN: <1GB free, model downloads (~280MB) may fail")

    # Optional models (do not download, just check cache)
    try:
        import spacy  # noqa: F401

        print("  spacy: installed")
    except ImportError:
        print("  spacy: not installed (run: pip install spacy; python -m spacy download en_core_web_sm)")

    if ok:
        print("\ndoctor: all essential checks passed ✓")
        print("Next: python -m adonis.web  or  adonis  → http://127.0.0.1:8000/")
    else:
        print("\ndoctor: some checks failed — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
