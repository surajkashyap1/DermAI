# DermAI

DermAI is a full-stack dermatology web app with:

- a chat-first skin cancer assistant
- grounded dermatology retrieval
- image upload for lesion analysis
- follow-up chat that uses uploaded image context

## Stack

- Frontend: Next.js, TypeScript, Tailwind CSS
- Backend: FastAPI, LangGraph
- Retrieval: Qdrant, dense + sparse retrieval, reranking
- Vision: external skin-lesion model integrated from Hugging Face
- LLM: Groq-backed generation with local fallback behavior

## Current Product Surface

- single-page chat UI at `/`
- dermatology chat
- image upload and lesion classification
- multimodal follow-up in the same session
- dark-mode interface

## Repo Layout

```text
apps/
  api/        FastAPI backend
  web/        Next.js frontend
packages/
  shared/     Shared TypeScript contracts
services/
  ingestion/  Corpus ingestion and normalization
  retrieval/  Retrieval infrastructure
  eval/       Evaluation scripts
infra/
  compose/    Local container orchestration
  deploy/     Deployment notes and manifests
docs/         Architecture and API notes
```

## Requirements

- Node.js 20+
- Python 3.11
- Docker Desktop

## Environment

Create a repo-root `.env` file as needed. Common variables:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
DERMAI_GROQ_API_KEY=your_key_here
DERMAI_GROQ_MODEL=llama-3.1-8b-instant
DERMAI_QDRANT_URL=http://localhost:6333
DERMAI_VISION_MODEL_REPO_ID=devatreya/skin-lesion-resnet50
DERMAI_VISION_MODEL_FILENAME=resnet50_best.h5
DERMAI_VISION_MODEL_THRESHOLD=0.5
```

If `DERMAI_GROQ_API_KEY` is not set, the backend falls back to a deterministic local extractive answer mode.

## Install

### Frontend

```bash
npm install
```

### Backend

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r apps/api/requirements.txt
```

## Run Locally

Start Qdrant:

```bash
npm run infra:qdrant:up
```

Start the API:

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --app-dir apps/api
```

Start the web app:

```bash
npm run dev:web
```

Open:

- Web: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

## Notes

- The first image upload can be slower because the vision model is downloaded and cached locally.
- Qdrant should be running locally for the intended retrieval path.
- The app is chat-first, but image upload stays attached to the active session for follow-up questions.

## Useful Commands

```bash
npm run build:web
npm run typecheck:web
python services/eval/run_phase6_eval.py
python services/eval/run_retrieval_eval.py
```

## Deployment

Current deployment split:

- Web: Vercel
- API: Render

If redeploying:

1. push to `main`
2. let Vercel rebuild the web app
3. let Render rebuild the API Docker image
4. test chat, image upload, and image follow-up in production
