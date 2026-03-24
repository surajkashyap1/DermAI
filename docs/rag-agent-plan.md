# DermAI RAG Agent Plan

## Objective

Make DermAI's chat stack a production-style, agentic dermatology RAG system with:

- strong dermatology corpus coverage
- hybrid retrieval with reranking
- evidence-aware confidence behavior
- citation-grounded answers
- agentic retry and verification logic
- measurable retrieval and groundedness quality

This file is the execution plan for the remaining RAG work. It replaces ad hoc iteration.

## Current Status

Already implemented:

- source registry and compiled corpus pipeline
- markdown/text/PubMed-style JSON ingestion
- live PubMed fetch + normalize + registry sync path
- Qdrant-backed hybrid retrieval
- dense + sparse retrieval
- ColBERT-style reranking
- retrieval debug endpoints
- baseline LangGraph workflow
- initial confidence logic
- starter retrieval benchmark and eval runner

Current gaps:

- corpus breadth is still limited
- graph is structured but not yet fully agentic
- evidence sufficiency and citation verification are still weak
- evaluation is still light
- persistence/checkpointing is not yet used for graph state

## Design Principles

The remaining work should follow these principles:

1. Retrieval before generation. Improve evidence quality before prompt complexity.
2. Constrained agentic behavior. Use explicit graph nodes and bounded retries, not open-ended autonomous loops.
3. Measurable quality. Every major upgrade should add a benchmark, trace, or verification hook.
4. Safety by abstention. It is better to answer partially and cautiously than confidently exceed evidence.
5. Stable contracts. Improve internals without breaking the demo UI.

## Official Guidance Driving This Plan

LangGraph guidance we should follow:

- use stateful graphs for orchestration
- add persistence/checkpointing for durable execution and debugging
- keep graph execution deterministic and idempotent
- use subgraphs only when they improve reuse or separation of concerns

Qdrant guidance we should follow:

- use hybrid dense + sparse retrieval
- use late-interaction reranking for higher precision
- prefer multivector reranking setup for production-quality hybrid search
- keep payload metadata rich enough for filtering and diagnostics

## Remaining Workstreams

### Workstream 1: Corpus Expansion

Goal:
Make the corpus broad enough that common dermatology overview and comparison questions are well-supported.

Tasks:

- expand melanoma coverage with more review/guideline-style material
- expand non-melanoma coverage:
  - basal cell carcinoma
  - squamous cell carcinoma
  - actinic keratosis
  - benign nevi
  - atypical nevi
  - seborrheic keratosis
- add more clinician-oriented triage and follow-up material
- add more patient-explanation material, clearly tagged by audience
- enrich source metadata:
  - authority level
  - audience
  - disease tags
  - topic tags
  - section path
  - year
  - PMID / URL

Definition of done:

- broad overview questions return mixed-source coverage
- comparison questions consistently retrieve multiple relevant disease families
- retrieval benchmark includes at least 15 to 20 realistic dermatology cases

### Workstream 2: Retrieval Hardening

Goal:
Turn the current hybrid pipeline into a more production-grade retrieval system.

Tasks:

- align Qdrant server/client versions
- move from app-side rerank-only behavior toward a cleaner production retrieval path
- consider Qdrant multivector storage for late-interaction reranking
- add metadata filters for:
  - audience
  - authority level
  - source type
  - disease tags
- add retrieval score normalization and calibration review
- expose richer debug output:
  - query rewrite
  - candidate counts
  - stage-level scores
  - metadata coverage

Definition of done:

- local/dev retrieval consistently runs on `qdrant_hybrid`
- retrieval diagnostics clearly explain why a hit ranked where it did
- top-3 and top-5 relevance is stable across the benchmark set

### Workstream 3: Agentic Graph Upgrade

Goal:
Make the LangGraph flow properly agentic in a controlled, medical-safe way.

Tasks:

- add an evidence sufficiency node
- add a retrieval retry node when coverage is weak
- add bounded query decomposition for multi-part questions
- add a citation verification node
- add a response gate node:
  - answer
  - answer with limited-coverage framing
  - abstain
- refactor the graph into clearer subflows if needed:
  - routing
  - retrieval
  - verification
  - formatting

Definition of done:

- weak first-pass retrieval can trigger one bounded retry
- multi-part questions can be decomposed into explicit sub-queries
- unsupported answers are downgraded or refused before final output

### Workstream 4: Evidence-Controlled Generation

Goal:
Improve answer quality without sacrificing groundedness.

Tasks:

- pass structured evidence summaries into generation
- make answer style less repetitive and less safety-boilerplate-heavy
- separate:
  - evidence-backed claims
  - uncertainty statements
  - optional caution message
- add claim-to-citation verification heuristics
- reduce overuse of “see a dermatologist” unless the evidence actually implies escalation

Definition of done:

- broad answers feel complete when evidence is broad
- narrow answers stay concise
- low-confidence answers explain what is missing instead of sounding weak or generic

### Workstream 5: Persistence and Session Memory

Goal:
Use LangGraph the way it is meant to be used for durable, inspectable workflows.

Tasks:

- add a checkpointer for graph persistence
- tie graph runs to session or thread IDs
- persist graph state for debugging and replay
- add short-term memory trimming/summarization for longer conversations

Definition of done:

- graph runs can be inspected by thread
- state can be resumed or replayed without recomputing everything blindly
- long conversations do not degrade because of bloated raw history

### Workstream 6: Evaluation and Observability

Goal:
Measure the real quality of the RAG stack and tune it systematically.

Tasks:

- expand retrieval benchmark from starter set to a more realistic dermatology benchmark
- add groundedness checks
- add citation coverage checks
- add unsupported-claim checks
- record latency by graph stage
- capture query rewrite / retry / abstain statistics
- save evaluation artifacts for regression checking

Definition of done:

- there is a repeatable benchmark suite for retrieval and answer quality
- regressions are visible before deploy
- latency and quality tradeoffs are documented

## Execution Order

Do the remaining RAG work in this order:

1. Qdrant/version cleanup and retrieval hardening
2. corpus expansion batch 1
3. agentic graph upgrade: evidence sufficiency + retry
4. bounded query decomposition for multi-part questions
5. citation verification and response gate
6. persistence/checkpointer integration
7. larger evaluation suite and tuning
8. corpus expansion batch 2 if needed after benchmark results

## Concrete Next Tickets

These are the next tickets to execute one by one:

1. Align Qdrant Docker version with the Python client and verify clean startup. Completed.
2. Add evidence sufficiency node to the LangGraph flow. Completed.
3. Add one bounded retrieval retry path when evidence coverage is weak. Completed.
4. Add query decomposition for multi-part questions. Completed.
5. Add citation verification before final answer formatting. Completed.
6. Expand the retrieval benchmark to at least 12 queries. Completed.
7. Add another external-source ingestion batch focused on non-melanoma skin cancer. Completed.
8. Add graph persistence/checkpointing tied to session IDs. Completed.

## Definition of "Industry-Ready" For This Project

DermAI RAG is ready when it can:

- answer common dermatology and skin-cancer questions with good coverage
- compare melanoma and non-melanoma conditions without collapsing into one disease family
- abstain or narrow scope when evidence is weak
- show citations that actually support the answer
- recover from weak retrieval with bounded retries or decomposition
- be debugged and benchmarked systematically

It does not need to be a research-perfect medical system. It does need to be a strong, transparent, evidence-first portfolio system.

## Sources

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph durable execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
- LangGraph subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Qdrant hybrid search with reranking: https://qdrant.tech/documentation/search-precision/reranking-hybrid-search/
- Qdrant multivectors and late interaction: https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/
