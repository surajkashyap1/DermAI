# Ingestion

DermAI now uses a registry-driven ingestion flow for the local RAG foundation.

## Source Registry

The source registry lives at:

```text
services/ingestion/data/source-registry.json
```

Each entry declares:

- source id
- source path
- loader type
- enabled flag
- metadata overrides

Current loader types:

- `structured_text` for markdown and text fixtures
- `pubmed_json` for normalized PubMed-style JSON fixtures

## External Source Workflow

PubMed-style ingestion now has three explicit stages:

1. fetch raw search results
2. normalize them into registry-ready JSON source files
3. sync those normalized files into the source registry

Example:

```bash
python services/ingestion/fetch_pubmed.py --dataset-id melanoma-basics --query "melanoma early recognition risk factors" --retmax 10
python services/ingestion/normalize_pubmed.py --dataset-id melanoma-basics --topic-tags "melanoma,risk factors" --disease-tags "melanoma,skin cancer"
python services/ingestion/sync_pubmed_registry.py --dataset-id melanoma-basics --topic-tags "melanoma,risk factors" --disease-tags "melanoma,skin cancer"
python services/ingestion/build_corpus.py
```

Output locations:

- raw fetched payloads: `services/ingestion/data/raw/pubmed`
- normalized registry-ready files: `services/ingestion/data/normalized/pubmed`
- compiled RAG corpus: `services/ingestion/data/compiled/corpus.json`

## Build the local corpus index

```bash
python services/ingestion/build_corpus.py
```

This reads the registry, loads the enabled sources, and writes a compiled corpus file to:

```text
services/ingestion/data/compiled/corpus.json
```

The compiled corpus now includes:

- document records
- chunk records
- source type
- authority level
- audience
- disease tags
- topic tags
- section path

This is still a local foundation rather than the final medical corpus, but it is now shaped for the next step: real external document ingestion and stronger retrieval infrastructure.
