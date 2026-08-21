.PHONY: install ingest pipeline report eval serve test lint typecheck clean

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

ingest:
	python scripts/ingest_corpus.py --in data/corpus/sample/

pipeline:
	python scripts/run_pipeline.py

report:
	python scripts/render_report.py

eval:
	python scripts/eval_report.py

serve:
	python -m adonis.web

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src/adonis

clean:
	rm -rf data/db/adonis.sqlite reports/*.html .pytest_cache .ruff_cache .mypy_cache
