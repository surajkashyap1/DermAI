# DermAI Architecture

## Objective

DermAI is a full-stack dermatology product with two major capability tracks:

- evidence-grounded dermatology chat
- lesion image analysis with explainability

Phases 1 through 6 establish the current platform shell, grounded chat experience, demo vision flow, and multimodal session behavior.

## Repository Model

- `apps/web`: shareable Next.js website and demo application
- `apps/api`: FastAPI backend with typed HTTP contracts
- `packages/shared`: shared TypeScript contracts consumed by the web app
- `services/ingestion`: future corpus loading and indexing pipeline
- `services/retrieval`: future hybrid retrieval and reranking layer
- `services/vision`: future classifier and Grad-CAM logic
- `services/eval`: local benchmark runners and diagnostics
- `infra`: local compose and deployment scaffolding

## Current Runtime Flow

1. The web app renders the landing page and demo shell.
2. The demo shell reads backend health and version information.
3. The API loads a compiled local dermatology corpus.
4. A LangGraph workflow classifies the request, rewrites weak queries, and routes greetings or emergency prompts appropriately.
5. The retrieval service ranks relevant chunks and maps them to citations.
6. The provider layer uses Groq when configured and falls back to a local extractive mode otherwise.
7. The vision service validates uploads, computes deterministic heuristic image metrics, and returns an overlay preview plus cautious result metadata.
8. Image uploads are attached to the active session, and follow-up questions can use that image result as non-diagnostic context for retrieval and answer generation.
9. Request IDs and route-level logs make chat and upload failures traceable during local development and demo use.
10. Later work will upgrade retrieval, orchestration, and the vision model without breaking the core contract shape.

## Planned RAG Upgrades

Later phases will implement this workflow:

1. classify user intent
2. optionally rewrite the query
3. run hybrid retrieval
4. rerank evidence
5. generate a grounded answer
6. verify citation support
7. apply confidence gate and safety logic
8. stream the formatted response back to the client

## Deployment Direction

The repo includes starter Dockerfiles and a compose file for local orchestration. Full production deployment, vector storage, and database wiring still come later.

## Python Runtime

DermAI targets Python 3.11 for backend work. This is the intended baseline for FastAPI, LangGraph, and later ML-oriented dependencies.
