FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.hf_cache \
    USE_TF=0

# System deps for pillow/faiss wheels are already covered by the slim image;
# install Python deps first for better layer caching.
COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Engine + assets
COPY dermai ./dermai
COPY models ./models
COPY data ./data
COPY apps/api ./apps/api

# Pre-build the FAISS index (and cache the embedding model inside the image),
# so the first request doesn't pay indexing/download latency.
RUN python -m dermai.rag.index

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port ${PORT:-8000}"]
