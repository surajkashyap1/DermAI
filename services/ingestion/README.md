# Ingestion

Phase 2 ships a seed dermatology corpus for local development.

## Build the local corpus index

```bash
python services/ingestion/build_corpus.py
```

This reads markdown sources from `services/ingestion/data/sources` and writes a compiled corpus file to `services/ingestion/data/compiled/corpus.json`.

The compiled corpus is intentionally local and lightweight. It is a bridge to later phases where vector infrastructure and richer metadata will be added.
