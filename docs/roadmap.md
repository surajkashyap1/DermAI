# DermAI Roadmap

## Phase 1

- monorepo reset
- Next.js website shell
- FastAPI skeleton
- shared contracts
- docs and conventions

## Phase 2

- ingestion pipeline
- metadata schema
- vector storage integration
- first cited QA endpoint

## Phase 3

- LangGraph chat workflow
- hybrid retrieval and reranking
- confidence gate
- streaming responses

## Phase 4

- image classifier endpoint
- upload workflow
- Grad-CAM overlays

## Phase 5

- combined image plus chat sessions
- multimodal answer formatting

## Phase 6

- evaluation harness
- tracing and diagnostics
- Docker hardening
- deployment polish

## Current Status

- Phases 1 through 6 are implemented for the current demo build.
- Grounded dermatology chat is real, but the corpus and retrieval stack are still early-stage.
- Vision upload and multimodal follow-up are real product flows, but the vision model is still heuristic rather than trained.
- The next major upgrade track is quality: stronger corpus ingestion, better retrieval, and a real lesion classifier.
