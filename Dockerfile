FROM ghcr.io/astral-sh/uv:0.11.3 AS uv

FROM python:3.11-slim

COPY --from=uv /uv /uvx /bin/

ARG GIT_COMMIT=unknown
ARG GIT_TREE_STATE=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    GIT_COMMIT=${GIT_COMMIT} \
    GIT_TREE_STATE=${GIT_TREE_STATE} \
    UV_VERSION=0.11.3 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uvicorn", "hybrid_retrieval_lab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
