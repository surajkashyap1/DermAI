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
5. The ingestion layer now uses an explicit source registry, richer metadata, and typed loaders for structured text and PubMed-style JSON sources.
6. The retrieval service compiles that corpus and uses a hybrid-ready dense plus sparse retrieval path, followed by a ColBERT-style rerank pass, with lexical fallback when hybrid search is unavailable.
7. Retrieval debug endpoints expose backend status plus dense hits, sparse hits, merged candidates, and reranked results so query quality can be inspected directly during development.
8. The provider layer uses Groq when configured and falls back to a local extractive mode otherwise.
9. The vision service validates uploads, computes deterministic heuristic image metrics, and returns an overlay preview plus cautious result metadata.
10. Image uploads are attached to the active session, and follow-up questions can use that image result as non-diagnostic context for retrieval and answer generation.
11. Request IDs and route-level logs make chat and upload failures traceable during local development and demo use.
12. Later work will replace the seed registry with broader external ingestion and a real vision model without breaking the core contract shape.

## Planned RAG Upgrades

Later phases will implement this workflow:

1. classify user intent
2. optionally rewrite the query
3. run hybrid retrieval against a richer corpus and Qdrant-backed index
4. rerank evidence with a stronger dedicated reranker
5. generate a grounded answer
6. verify citation support
7. apply confidence gate and safety logic
8. stream the formatted response back to the client

## Deployment Direction

The repo includes starter Dockerfiles and a compose file for local orchestration. Qdrant wiring is now present for local and future service-backed retrieval, and local development should prefer a running Qdrant service over the embedded file store to avoid lock contention.

## Python Runtime

DermAI targets Python 3.11 for backend work. This is the intended baseline for FastAPI, LangGraph, and later ML-oriented dependencies.
