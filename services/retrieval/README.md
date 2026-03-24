# Retrieval

Phase 2 uses a lightweight local retrieval pipeline:

- compiled chunk corpus from `services/ingestion`
- lexical token scoring with metadata-aware ranking
- citation mapping by chunk ID

This is intentionally simple and deterministic. It gives the app grounded answers now and leaves room for Qdrant and reranking in later phases.
