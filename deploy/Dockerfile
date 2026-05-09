FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (cached layer — only re-runs when pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project --no-dev

# Install application packages
COPY finbot/ ./finbot/
COPY rag_tool/ ./rag_tool/
RUN uv sync --no-dev

# Writable dirs for ephemeral data (disk cache + RAG ingestion staging)
RUN mkdir -p /app/rag_tool/documents /app/rag_tool/data/index /app/.finbot_cache

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# --workers 1: InMemorySessionService is per-process; multiple workers break session continuity
CMD ["uv", "run", "uvicorn", "finbot.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop"]
