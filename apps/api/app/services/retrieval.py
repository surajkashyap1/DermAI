from __future__ import annotations

import atexit
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.services.corpus import load_compiled_corpus, project_root


logger = logging.getLogger("dermai.api.retrieval")
INDEX_SCHEMA_VERSION = "hybrid-v1"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "what",
    "when",
    "with",
}

QUERY_EXPANSIONS = {
    "basal cell carcinoma": "bcc non melanoma skin cancer pearly papule rolled borders",
    "bcc": "basal cell carcinoma non melanoma skin cancer pearly papule",
    "squamous cell carcinoma": "scc non melanoma skin cancer scaly hyperkeratotic crusted lesion",
    "scc": "squamous cell carcinoma non melanoma skin cancer scaly crusted lesion",
    "self-monitoring": "surveillance follow-up changing lesion dermatologist review",
    "warning signs": "red flags evolution bleeding change irregular borders asymmetry",
    "different from melanoma": "differential benign nevus seborrheic keratosis basal cell carcinoma squamous cell carcinoma",
}


@dataclass
class RetrievalHit:
    id: str
    title: str
    source: str
    section: str
    snippet: str
    href: str
    score: float
    text: str
    year: int
    topic_tags: list[str]
    disease_tags: list[str]
    source_type: str
    authority_level: str
    audience: str
    pmid: str | None
    section_path: list[str]


@dataclass
class RetrievalDebugStageHit:
    id: str
    title: str
    source: str
    section: str
    snippet: str
    href: str
    score: float
    stage_score: float
    text: str
    year: int
    topic_tags: list[str]
    disease_tags: list[str]
    source_type: str
    authority_level: str
    audience: str
    pmid: str | None
    section_path: list[str]


@dataclass
class RetrievalCandidate:
    payload: dict[str, Any]
    dense_score: float
    sparse_score: float
    merged_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0


class LexicalFallbackRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._document_frequency = self._build_document_frequency()
        self._chunk_count = len(self._chunks)
        self._reranker = None

    def _build_document_frequency(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for chunk in self._chunks:
            counter.update(chunk["token_counts"].keys())
        return counter

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in STOPWORDS
        ]

    def expand_query(self, query: str) -> str:
        expanded = query
        lowered = query.lower()
        for pattern, addition in QUERY_EXPANSIONS.items():
            if pattern in lowered and addition not in expanded:
                expanded = f"{expanded} {addition}"
        return expanded

    def _idf(self, token: str) -> float:
        doc_freq = self._document_frequency.get(token, 0)
        return math.log((self._chunk_count + 1) / (doc_freq + 1)) + 1.0

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_tokens = self._tokenize(self.expand_query(query))
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        hits: list[RetrievalHit] = []
        for chunk in self._chunks:
            token_counts = chunk["token_counts"]
            overlap = set(query_counts).intersection(token_counts)
            if not overlap:
                continue

            score = 0.0
            for token in overlap:
                score += min(query_counts[token], token_counts[token]) * self._idf(token)

            normalized = score / math.sqrt(max(len(token_counts), 1))
            tag_bonus = 0.25 if overlap.intersection(set(chunk.get("topic_tags") or [])) else 0.0
            disease_bonus = 0.25 if overlap.intersection(set(chunk.get("disease_tags") or [])) else 0.0
            final_score = round(normalized + tag_bonus + disease_bonus, 4)
            hits.append(self._payload_to_hit(chunk, final_score))

        hits.sort(key=lambda item: item.score, reverse=True)
        expanded_query = self.expand_query(query)
        candidate_limit = max(top_k * settings.retrieval_candidate_multiplier, settings.retrieval_candidate_min)
        reranked_hits, _ = self._rerank_hits(expanded_query, hits[:candidate_limit])
        return reranked_hits[:top_k]

    def _payload_to_hit(self, payload: dict[str, Any], score: float) -> RetrievalHit:
        return RetrievalHit(
            id=payload["id"],
            title=payload["title"],
            source=payload["source"],
            section=payload["section"],
            snippet=payload["snippet"],
            href=payload["href"],
            score=score,
            text=payload["text"],
            year=payload["year"],
            topic_tags=payload.get("topic_tags") or [],
            disease_tags=payload.get("disease_tags") or [],
            source_type=payload.get("source_type", "unknown"),
            authority_level=payload.get("authority_level", "unknown"),
            audience=payload.get("audience", "clinician"),
            pmid=payload.get("pmid"),
            section_path=payload.get("section_path") or [payload["title"], payload["section"]],
        )

    def _reranker_model(self) -> LateInteractionTextEmbedding | None:
        if not settings.retrieval_reranker_enabled:
            return None
        if self._reranker is None:
            self._reranker = LateInteractionTextEmbedding(model_name=settings.retrieval_reranker_model)
        return self._reranker

    def _rerank_hits(self, expanded_query: str, hits: list[RetrievalHit]) -> tuple[list[RetrievalHit], list[RetrievalDebugStageHit]]:
        reranker = self._reranker_model()
        if reranker is None or not hits:
            debug_hits = [
                RetrievalDebugStageHit(
                    id=hit.id,
                    title=hit.title,
                    source=hit.source,
                    section=hit.section,
                    snippet=hit.snippet,
                    href=hit.href,
                    score=hit.score,
                    stage_score=hit.score,
                    text=hit.text,
                    year=hit.year,
                    topic_tags=hit.topic_tags,
                    disease_tags=hit.disease_tags,
                    source_type=hit.source_type,
                    authority_level=hit.authority_level,
                    audience=hit.audience,
                    pmid=hit.pmid,
                    section_path=hit.section_path,
                )
                for hit in hits
            ]
            return hits, debug_hits

        query_embedding = next(reranker.query_embed([expanded_query]))
        passage_embeddings = list(reranker.passage_embed(hit.text for hit in hits))
        scored_hits: list[tuple[RetrievalHit, float]] = []
        for hit, passage_embedding in zip(hits, passage_embeddings, strict=True):
            rerank_score = float(np.max(query_embedding @ passage_embedding.T, axis=1).sum())
            scored_hits.append((hit, rerank_score))

        lexical_rank = {hit.id: rank for rank, hit in enumerate(hits, start=1)}
        reranked_order = sorted(scored_hits, key=lambda item: item[1], reverse=True)
        rerank_rank = {hit.id: rank for rank, (hit, _) in enumerate(reranked_order, start=1)}
        scored_final_hits: list[tuple[RetrievalHit, float]] = []
        for hit, rerank_score in reranked_order:
            final_score = round(
                (1.0 / (50 + lexical_rank[hit.id]))
                + (1.0 / (15 + rerank_rank[hit.id]))
                + hit.score * 0.2,
                4,
            )
            reranked_hit = RetrievalHit(
                id=hit.id,
                title=hit.title,
                source=hit.source,
                section=hit.section,
                snippet=hit.snippet,
                href=hit.href,
                score=final_score,
                text=hit.text,
                year=hit.year,
                topic_tags=hit.topic_tags,
                disease_tags=hit.disease_tags,
                source_type=hit.source_type,
                authority_level=hit.authority_level,
                audience=hit.audience,
                pmid=hit.pmid,
                section_path=hit.section_path,
            )
            scored_final_hits.append((reranked_hit, rerank_score))

        scored_final_hits.sort(key=lambda item: item[0].score, reverse=True)
        final_hits = [hit for hit, _ in scored_final_hits]
        debug_hits = [
            RetrievalDebugStageHit(
                id=reranked_hit.id,
                title=reranked_hit.title,
                source=reranked_hit.source,
                section=reranked_hit.section,
                snippet=reranked_hit.snippet,
                href=reranked_hit.href,
                score=reranked_hit.score,
                stage_score=round(rerank_score, 4),
                text=reranked_hit.text,
                year=reranked_hit.year,
                topic_tags=reranked_hit.topic_tags,
                disease_tags=reranked_hit.disease_tags,
                source_type=reranked_hit.source_type,
                authority_level=reranked_hit.authority_level,
                audience=reranked_hit.audience,
                pmid=reranked_hit.pmid,
                section_path=reranked_hit.section_path,
            )
            for reranked_hit, rerank_score in scored_final_hits
        ]
        return final_hits, debug_hits

    def debug_search(self, query: str, top_k: int) -> dict[str, Any]:
        expanded_query = self.expand_query(query)
        candidate_limit = max(top_k * settings.retrieval_candidate_multiplier, settings.retrieval_candidate_min)
        lexical_hits = self.search(query, candidate_limit)
        reranked_hits, reranked_debug_hits = self._rerank_hits(expanded_query, lexical_hits)
        lexical_debug_hits = [
            RetrievalDebugStageHit(
                id=hit.id,
                title=hit.title,
                source=hit.source,
                section=hit.section,
                snippet=hit.snippet,
                href=hit.href,
                score=hit.score,
                stage_score=hit.score,
                text=hit.text,
                year=hit.year,
                topic_tags=hit.topic_tags,
                disease_tags=hit.disease_tags,
                source_type=hit.source_type,
                authority_level=hit.authority_level,
                audience=hit.audience,
                pmid=hit.pmid,
                section_path=hit.section_path,
            )
            for hit in lexical_hits[:top_k]
        ]
        return {
            "query": query,
            "expanded_query": expanded_query,
            "backend": "lexical_fallback",
            "candidate_count": len(lexical_hits),
            "dense_hits": [],
            "sparse_hits": [],
            "merged_hits": lexical_debug_hits,
            "reranked_hits": reranked_debug_hits[:top_k],
            "hits": reranked_hits[:top_k],
        }


class QdrantHybridRetriever:
    def __init__(self, corpus: dict[str, Any]) -> None:
        self._corpus = corpus
        self._chunks = corpus["chunks"]
        self._dense = TextEmbedding(model_name=settings.retrieval_dense_model)
        self._sparse = SparseTextEmbedding(model_name=settings.retrieval_sparse_model)
        self._reranker = None
        self._client = self._create_client()
        self._index_ready = False
        self._state_path = project_root() / "generated" / "qdrant" / "index-state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_client(self) -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )

        return QdrantClient(path=str(project_root() / "generated" / "qdrant" / "local-store"))

    def close(self) -> None:
        self._client.close()

    def _corpus_signature(self) -> str:
        return f"{self._corpus['version']}:{self._corpus['chunkCount']}"

    def expand_query(self, query: str) -> str:
        expanded = query
        lowered = query.lower()
        for pattern, addition in QUERY_EXPANSIONS.items():
            if pattern in lowered and addition not in expanded:
                expanded = f"{expanded} {addition}"
        return expanded

    def _load_state(self) -> dict[str, Any] | None:
        if not self._state_path.exists():
            return None
        return json.loads(self._state_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        state = {
            "index_schema": INDEX_SCHEMA_VERSION,
            "corpus_signature": self._corpus_signature(),
            "dense_model": settings.retrieval_dense_model,
            "sparse_model": settings.retrieval_sparse_model,
            "reranker_model": settings.retrieval_reranker_model,
        }
        self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _ensure_index(self) -> None:
        if self._index_ready:
            return

        state = self._load_state()
        expected = {
            "index_schema": INDEX_SCHEMA_VERSION,
            "corpus_signature": self._corpus_signature(),
            "dense_model": settings.retrieval_dense_model,
            "sparse_model": settings.retrieval_sparse_model,
            "reranker_model": settings.retrieval_reranker_model,
        }

        collection_exists = self._client.collection_exists(settings.qdrant_collection)
        if collection_exists and state == expected:
            self._index_ready = True
            return

        logger.info("Rebuilding hybrid retrieval index for corpus signature %s", expected["corpus_signature"])
        self._client.recreate_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )

        dense_vectors = [vector.tolist() for vector in self._dense.embed(chunk["text"] for chunk in self._chunks)]
        sparse_vectors = list(self._sparse.embed(chunk["text"] for chunk in self._chunks))
        points: list[models.PointStruct] = []

        for point_id, (chunk, dense_vector, sparse_vector) in enumerate(
            zip(self._chunks, dense_vectors, sparse_vectors, strict=True),
            start=1,
        ):
            payload = {key: value for key, value in chunk.items() if key != "token_counts"}
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        self._client.upsert(settings.qdrant_collection, points=points, wait=True)
        self._save_state()
        self._index_ready = True

    def _query_dense(self, query: str, limit: int) -> list[models.ScoredPoint]:
        vector = next(self._dense.embed([query])).tolist()
        response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _query_sparse(self, query: str, limit: int) -> list[models.ScoredPoint]:
        vector = next(self._sparse.embed([query]))
        response = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=models.SparseVector(
                indices=vector.indices.tolist(),
                values=vector.values.tolist(),
            ),
            using="sparse",
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _reranker_model(self) -> LateInteractionTextEmbedding | None:
        if not settings.retrieval_reranker_enabled:
            return None
        if self._reranker is None:
            self._reranker = LateInteractionTextEmbedding(model_name=settings.retrieval_reranker_model)
        return self._reranker

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in STOPWORDS
        }

    def _metadata_bonus(
        self,
        payload: dict[str, Any],
        query_tokens: set[str],
        expanded_query: str,
    ) -> float:
        title_tokens = self._tokenize(payload.get("title", ""))
        topic_tokens = self._tokenize(" ".join(payload.get("topic_tags") or []))
        disease_tokens = self._tokenize(" ".join(payload.get("disease_tags") or []))
        section_tokens = self._tokenize(" ".join(payload.get("section_path") or []))
        query_text = expanded_query.lower()
        phrase_bonus = 0.0
        for phrase in [*(payload.get("disease_tags") or []), *(payload.get("topic_tags") or [])]:
            normalized = phrase.lower()
            if normalized and normalized in query_text:
                phrase_bonus += 0.1

        title_bonus = 0.15 if query_tokens.intersection(title_tokens) else 0.0
        topic_bonus = 0.12 if query_tokens.intersection(topic_tokens) else 0.0
        disease_bonus = 0.12 if query_tokens.intersection(disease_tokens) else 0.0
        section_bonus = 0.08 if query_tokens.intersection(section_tokens) else 0.0
        authority_bonus = 0.05 if payload.get("authority_level") != "internal_seed" else 0.0
        return round(
            title_bonus + topic_bonus + disease_bonus + section_bonus + authority_bonus + phrase_bonus,
            4,
        )

    def _candidate_to_hit(self, candidate: RetrievalCandidate) -> RetrievalHit:
        payload = candidate.payload
        return RetrievalHit(
            id=payload["id"],
            title=payload["title"],
            source=payload["source"],
            section=payload["section"],
            snippet=payload["snippet"],
            href=payload["href"],
            score=round(candidate.final_score, 4),
            text=payload["text"],
            year=payload["year"],
            topic_tags=payload.get("topic_tags") or [],
            disease_tags=payload.get("disease_tags") or [],
            source_type=payload.get("source_type", "unknown"),
            authority_level=payload.get("authority_level", "unknown"),
            audience=payload.get("audience", "clinician"),
            pmid=payload.get("pmid"),
            section_path=payload.get("section_path") or [payload["title"], payload["section"]],
        )

    def _candidate_to_debug_hit(self, candidate: RetrievalCandidate, stage_score: float) -> RetrievalDebugStageHit:
        payload = candidate.payload
        return RetrievalDebugStageHit(
            id=payload["id"],
            title=payload["title"],
            source=payload["source"],
            section=payload["section"],
            snippet=payload["snippet"],
            href=payload["href"],
            score=round(candidate.final_score, 4),
            stage_score=round(stage_score, 4),
            text=payload["text"],
            year=payload["year"],
            topic_tags=payload.get("topic_tags") or [],
            disease_tags=payload.get("disease_tags") or [],
            source_type=payload.get("source_type", "unknown"),
            authority_level=payload.get("authority_level", "unknown"),
            audience=payload.get("audience", "clinician"),
            pmid=payload.get("pmid"),
            section_path=payload.get("section_path") or [payload["title"], payload["section"]],
        )

    def _rerank_candidates(self, expanded_query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        reranker = self._reranker_model()
        if reranker is None or not candidates:
            for candidate in candidates:
                candidate.final_score = candidate.merged_score
            return candidates

        query_embedding = next(reranker.query_embed([expanded_query]))
        passage_embeddings = list(
            reranker.passage_embed(candidate.payload["text"] for candidate in candidates)
        )
        for candidate, passage_embedding in zip(candidates, passage_embeddings, strict=True):
            candidate.rerank_score = float(np.max(query_embedding @ passage_embedding.T, axis=1).sum())

        merged_rank = {candidate.payload["id"]: rank for rank, candidate in enumerate(candidates, start=1)}
        reranked = sorted(candidates, key=lambda item: item.rerank_score, reverse=True)
        rerank_rank = {candidate.payload["id"]: rank for rank, candidate in enumerate(reranked, start=1)}
        for candidate in reranked:
            candidate.final_score = round(
                (1.0 / (50 + merged_rank[candidate.payload["id"]]))
                + (1.0 / (15 + rerank_rank[candidate.payload["id"]]))
                + candidate.merged_score * 0.15,
                4,
            )
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked

    def _search_candidates(self, query: str, top_k: int) -> dict[str, Any]:
        self._ensure_index()
        expanded_query = self.expand_query(query)
        candidate_limit = max(top_k * settings.retrieval_candidate_multiplier, settings.retrieval_candidate_min)
        dense_points = self._query_dense(expanded_query, candidate_limit)
        sparse_points = self._query_sparse(expanded_query, candidate_limit)

        dense_rank = {str(point.payload["id"]): index for index, point in enumerate(dense_points, start=1)}
        sparse_rank = {str(point.payload["id"]): index for index, point in enumerate(sparse_points, start=1)}
        dense_score = {str(point.payload["id"]): float(point.score or 0.0) for point in dense_points}
        sparse_score = {str(point.payload["id"]): float(point.score or 0.0) for point in sparse_points}
        payloads = {
            str(point.payload["id"]): point.payload
            for point in [*dense_points, *sparse_points]
            if point.payload is not None
        }
        all_ids = set(dense_rank) | set(sparse_rank)
        query_tokens = self._tokenize(expanded_query)

        candidates: list[RetrievalCandidate] = []
        for point_id in all_ids:
            payload = payloads[point_id]
            rrf = 0.0
            if point_id in dense_rank:
                rrf += 1.0 / (60 + dense_rank[point_id])
            if point_id in sparse_rank:
                rrf += 1.0 / (60 + sparse_rank[point_id])

            base_score = round(
                rrf
                + min(dense_score.get(point_id, 0.0), 1.0) * 0.2
                + min(sparse_score.get(point_id, 0.0), 4.0) * 0.05,
                4,
            )
            merged_score = round(
                base_score + self._metadata_bonus(payload, query_tokens, expanded_query),
                4,
            )
            candidates.append(
                RetrievalCandidate(
                    payload=payload,
                    dense_score=round(dense_score.get(point_id, 0.0), 4),
                    sparse_score=round(sparse_score.get(point_id, 0.0), 4),
                    merged_score=merged_score,
                    final_score=merged_score,
                )
            )

        merged_candidates = sorted(candidates, key=lambda item: item.merged_score, reverse=True)
        reranked_candidates = self._rerank_candidates(expanded_query, merged_candidates[:candidate_limit])
        final_hits = [self._candidate_to_hit(candidate) for candidate in reranked_candidates[:top_k]]
        return {
            "query": query,
            "expanded_query": expanded_query,
            "candidate_count": len(merged_candidates),
            "dense_hits": sorted(
                [candidate for candidate in merged_candidates if candidate.dense_score > 0],
                key=lambda item: item.dense_score,
                reverse=True,
            )[:top_k],
            "sparse_hits": sorted(
                [candidate for candidate in merged_candidates if candidate.sparse_score > 0],
                key=lambda item: item.sparse_score,
                reverse=True,
            )[:top_k],
            "merged_hits": merged_candidates[:top_k],
            "reranked_hits": reranked_candidates[:top_k],
            "hits": final_hits,
        }

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        return self._search_candidates(query, top_k)["hits"]

    def debug_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        result = self._search_candidates(query, top_k)
        return {
            "query": result["query"],
            "expanded_query": result["expanded_query"],
            "backend": "qdrant_hybrid",
            "candidate_count": result["candidate_count"],
            "dense_hits": [self._candidate_to_debug_hit(candidate, candidate.dense_score) for candidate in result["dense_hits"]],
            "sparse_hits": [self._candidate_to_debug_hit(candidate, candidate.sparse_score) for candidate in result["sparse_hits"]],
            "merged_hits": [self._candidate_to_debug_hit(candidate, candidate.merged_score) for candidate in result["merged_hits"]],
            "reranked_hits": [
                self._candidate_to_debug_hit(
                    candidate,
                    candidate.rerank_score if candidate.rerank_score > 0 else candidate.merged_score,
                )
                for candidate in result["reranked_hits"]
            ],
            "hits": result["hits"],
        }


class RetrievalService:
    _shared_backend: QdrantHybridRetriever | None = None
    _shared_backend_failed: bool = False

    def __init__(self) -> None:
        self._corpus = load_compiled_corpus()
        self._chunks = self._corpus["chunks"]
        self._fallback = LexicalFallbackRetriever(self._chunks)
        if RetrievalService._shared_backend is not None:
            self._backend = RetrievalService._shared_backend
            return

        if RetrievalService._shared_backend_failed:
            self._backend = None
            return

        try:
            backend = QdrantHybridRetriever(self._corpus)
            atexit.register(backend.close)
            RetrievalService._shared_backend = backend
            self._backend = backend
        except Exception as error:
            logger.warning("Hybrid retriever unavailable, using lexical fallback: %s", error)
            RetrievalService._shared_backend_failed = True
            self._backend = None

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        if self._backend is None:
            return self._fallback.search(query, top_k)

        try:
            return self._backend.search(query, top_k)
        except Exception as error:
            logger.warning("Hybrid search failed, falling back to lexical retrieval: %s", error)
            return self._fallback.search(query, top_k)

    def debug_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if self._backend is None:
            return self._fallback.debug_search(query, top_k)

        try:
            return self._backend.debug_search(query, top_k)
        except Exception as error:
            logger.warning("Hybrid debug search failed, falling back to lexical retrieval: %s", error)
            return self._fallback.debug_search(query, top_k)

    def backend_name(self) -> str:
        return "qdrant_hybrid" if self._backend is not None else "lexical_fallback"

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name(),
            "qdrant_url": settings.qdrant_url or None,
            "qdrant_collection": settings.qdrant_collection,
            "reranker_enabled": settings.retrieval_reranker_enabled,
            "reranker_model": settings.retrieval_reranker_model if settings.retrieval_reranker_enabled else None,
            "corpus_documents": self._corpus["documentCount"],
            "corpus_chunks": self._corpus["chunkCount"],
        }

    def citation_by_id(self, citation_id: str) -> RetrievalHit | None:
        chunk = next((item for item in self._chunks if item["id"] == citation_id), None)
        if not chunk:
            return None

        return RetrievalHit(
            id=chunk["id"],
            title=chunk["title"],
            source=chunk["source"],
            section=chunk["section"],
            snippet=chunk["snippet"],
            href=chunk["href"],
            score=0.0,
            text=chunk["text"],
            year=chunk["year"],
            topic_tags=chunk.get("topic_tags") or [],
            disease_tags=chunk.get("disease_tags") or [],
            source_type=chunk.get("source_type", "unknown"),
            authority_level=chunk.get("authority_level", "unknown"),
            audience=chunk.get("audience", "clinician"),
            pmid=chunk.get("pmid"),
            section_path=chunk.get("section_path") or [chunk["title"], chunk["section"]],
        )
