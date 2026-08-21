# Adonis — standalone container
# Build: docker build -t adonis:0.1.0 .
# Run:   docker run --rm -p 8000:8000 -v $PWD/data:/app/data -v $PWD/reports:/app/reports adonis:0.1.0
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for faiss + docx (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md PLAN.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

RUN pip install --upgrade pip \
    && pip install -e . \
    && python -m spacy download en_core_web_sm || echo "spacy model optional at build"

# Pre-create dirs (volumes may overlay)
RUN mkdir -p data/corpus/sample data/db reports

EXPOSE 8000

# Healthcheck hits the console
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/api/status', timeout=3).raise_for_status()" || exit 1

VOLUME ["/app/data", "/app/reports"]

CMD ["python", "-m", "adonis.web"]
