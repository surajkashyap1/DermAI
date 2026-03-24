# Retrieval

The retrieval stack is now moving toward a hybrid setup:

- compiled chunk corpus from `services/ingestion`
- dense embeddings with `BAAI/bge-small-en-v1.5`
- sparse retrieval with `Qdrant/bm25`
- Qdrant-compatible indexing and querying
- ColBERT-style reranking with `answerdotai/answerai-colbert-small-v1`
- metadata-aware score shaping after first-pass retrieval
- citation mapping by chunk ID

Recommended local setup:

1. `npm run infra:qdrant:up`
2. if you previously ran an older Qdrant image, `npm run infra:qdrant:refresh`
3. set `DERMAI_QDRANT_URL=http://localhost:6333` in the repo-root `.env`
4. restart the API
5. watch the API startup log and confirm it reports `Retrieval backend=qdrant_hybrid`

To inspect the local Qdrant container:

- `npm run infra:qdrant:ps`
- `npm run infra:qdrant:logs`

The current app still uses the local seed corpus, but retrieval is now structured so stronger document ingestion, Qdrant-backed indexing, and a dedicated reranker can be added without changing the chat contract.
