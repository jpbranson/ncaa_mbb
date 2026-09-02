# Serving image. Deliberately does NOT contain the 527 MB of training data:
# this image scores live games, it does not fit models. The ratings snapshot and
# the model artifact come in as a mounted volume, so swapping which model serves
# is a volume change and a restart, not a rebuild.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CBBWP_ROOT=/app \
    CBBWP_REGISTRY=/app/registry \
    CBBWP_LIVE_DIR=/data/live \
    CBBWP_API_HOST=0.0.0.0 \
    CBBWP_API_PORT=8808

WORKDIR /app

# libgomp is LightGBM's OpenMP runtime; the wheel needs it at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy polars lightgbm scikit-learn

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Run as a non-root user; the process only ever reads the registry and appends
# to one JSONL file.
RUN useradd --create-home --uid 10001 cbbwp \
 && mkdir -p /data/live && chown -R cbbwp /data
USER cbbwp

EXPOSE 8808
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8808/health || exit 1

ENTRYPOINT ["python3", "scripts/serve_live.py"]
