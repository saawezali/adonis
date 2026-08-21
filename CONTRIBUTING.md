# Contributing

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install   # pip install -e .[dev] + spacy model
make test      # 164 tests
make lint && make typecheck
python -m adonis.cli.doctor
```

## Workflow

1. `git checkout -b feat/your-feature`
2. Write tests first (`tests/unit/` or `tests/fakes.py` for LLM/NER/embedder fakes)
3. Keep `PLAN.md` as design record — update it if you change architecture, taxonomy, or schema; `README.md` is the user guide.
4. `make lint` (ruff) and `make typecheck` (mypy strict, 3.11) must pass.
5. Open PR with `make demo` screenshot/GIF if UI changed.

## Issue templates (use GitHub Issues)

* **Bug report:** repro steps, `adonis-doctor` output, `reports/eval.json` if eval-related.
* **Missed contradiction:** attach two claim texts + `recall@K` (`python scripts/measure_recall.py`) — helps tune `top_k`/`similarity_weight`.

## Labeling

```bash
python -m adonis.cli.label_pairs --pool entity      # contradictions
python -m adonis.cli.label_pairs --pool near_dup    # TN near_dup
python -m adonis.cli.label_pairs --pool unrelated   # TN unrelated
python scripts/eval_report.py  # → reports/eval.json
```

## Release

* Bump `adonis/__init__.py:__version__` + `pyproject.toml`, update `CHANGELOG.md`, tag `v0.1.0`, `make build`.
