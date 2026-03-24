# DermAI

DermAI is a full-stack dermatology demo product built as a portfolio-grade rebuild of the original conference project. The application combines a dermatology-focused evidence assistant with a lesion analysis workflow and is being rebuilt in phases.

## Current Scope
Phases 1 through 6 establish the current shareable demo state:

- Next.js web shell in `apps/web`
- FastAPI backend skeleton in `apps/api`
- shared frontend contracts in `packages/shared`
- local dermatology corpus ingestion in `services/ingestion`
- retrieval-backed chat with citations
- LangGraph-driven intent routing and response shaping
- demo heuristic image-analysis pipeline with overlay generation
- same-session image-plus-chat follow-up flow
- request tracing, safer error handling, and a local eval harness
- docs, repo conventions, and starter deployment scaffolding

## Repository Layout

```text
apps/
  api/        FastAPI backend
  web/        Next.js frontend
packages/
  shared/     Shared TypeScript contracts
services/
  ingestion/  Future ingestion pipeline
  retrieval/  Future retrieval system
  vision/     Future classifier and Grad-CAM services
  eval/       Future benchmarks and evaluation tooling
infra/
  compose/    Local container orchestration
  deploy/     Deployment notes and manifests
docs/         Architecture, API, roadmap
```

## Local Development

### Web

```bash
npm install
npm run dev:web
```

The web app runs on `http://localhost:3000`.

### API

DermAI should be run on **Python 3.11**. Create a Python 3.11 virtual environment, then install backend dependencies:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r apps/api/requirements.txt
python -m uvicorn app.main:app --reload --app-dir apps/api
```

The API runs on `http://localhost:8000`.

For backend local verification and tests:

```bash
pip install -r apps/api/requirements-dev.txt
```

If `py -3.11` is not available on your machine yet, install Python 3.11 first and recreate `.venv`. Python 3.14 works today, but it produces avoidable compatibility warnings with parts of the LangGraph stack.

## Groq Configuration

DermAI can use Groq as its hosted LLM provider. Add these variables to the repo-root `.env` or your shell when you are ready:

```bash
DERMAI_GROQ_API_KEY=your_key_here
DERMAI_GROQ_MODEL=llama-3.1-8b-instant
```

If no Groq key is configured, the backend falls back to a deterministic local extractive answer mode built from retrieved evidence.

## Current Endpoints

- `GET /health`
- `GET /version`
- `POST /chat`
- `POST /upload-image`
- `GET /citations/{citation_id}`
- `GET /session/{session_id}`

`POST /chat` runs against the local seed corpus and returns grounded citations. `POST /upload-image` returns a demo heuristic image-analysis payload with an overlay preview, and same-session follow-up questions can use that image result as non-diagnostic context. The current vision path is explicitly not a trained diagnostic model.

## Phase 6 Hardening

- API responses include an `x-request-id` header for debugging failed requests.
- Citation lookups now return real `404` errors instead of placeholder data.
- The demo UI surfaces upload and chat failures instead of silently degrading.
- A lightweight eval runner lives in `services/eval/run_phase6_eval.py`.

Run the eval harness from the repo root:

```bash
python services/eval/run_phase6_eval.py
```

## Deployment

DermAI is currently set up for:

- Vercel for the Next.js web app
- Render for the FastAPI API

Deployment files included in the repo:

- `render.yaml` for the API service
- `apps/api/Dockerfile` for Render's Docker deploy
- `infra/deploy/README.md` for environment and platform notes

Recommended production split:

- web: `your-domain.com`
- api: `api.your-domain.com`
