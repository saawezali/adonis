.PHONY: install ingest pipeline report eval serve test lint typecheck clean demo doctor build docker sample

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

sample:
	python scripts/generate_sample_corpus.py --out data/corpus/sample --force

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

doctor:
	python -m adonis.cli.doctor

demo: sample
	python scripts/ingest_corpus.py --in data/corpus/sample/
	python scripts/extract_claims.py
	python scripts/run_pipeline.py
	python scripts/render_report.py
	@echo "demo done — open reports/index.html and http://127.0.0.1:8000/ (make serve)"

build:
	python -m build
	twine check dist/*

docker:
	docker build -t adonis:0.1.0 .
	@echo "run: docker run --rm -p 8000:8000 -v $$PWD/data:/app/data -v $$PWD/reports:/app/reports adonis:0.1.0"

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src/adonis --ignore-missing-imports

clean:
	rm -rf data/db/adonis.sqlite reports/*.html reports/eval.json .pytest_cache .ruff_cache .mypy_cache dist/ build/
