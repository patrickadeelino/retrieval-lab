FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "hybrid_retrieval_lab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
