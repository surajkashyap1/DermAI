FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY apps/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY apps/api ./apps/api
COPY services/ingestion ./services/ingestion
RUN python services/ingestion/build_corpus.py
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port ${PORT:-8000}"]
