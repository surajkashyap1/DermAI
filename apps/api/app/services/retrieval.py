from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.services.corpus import load_compiled_corpus


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


class RetrievalService:
    def __init__(self) -> None:
        self._corpus = load_compiled_corpus()
        self._chunks = self._corpus["chunks"]
        self._document_frequency = self._build_document_frequency()
        self._chunk_count = len(self._chunks)

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

    def _idf(self, token: str) -> float:
        doc_freq = self._document_frequency.get(token, 0)
        return math.log((self._chunk_count + 1) / (doc_freq + 1)) + 1.0

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        query_tokens = self._tokenize(query)
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
            title_bonus = 0.35 if any(token in chunk["title"].lower() for token in overlap) else 0.0
            tag_bonus = 0.25 if any(token in " ".join(chunk["topic_tags"]).lower() for token in overlap) else 0.0
            final_score = round(normalized + title_bonus + tag_bonus, 4)

            hits.append(
                RetrievalHit(
                    id=chunk["id"],
                    title=chunk["title"],
                    source=chunk["source"],
                    section=chunk["section"],
                    snippet=chunk["snippet"],
                    href=chunk["href"],
                    score=final_score,
                    text=chunk["text"],
                    year=chunk["year"],
                    topic_tags=chunk["topic_tags"],
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def citation_by_id(self, citation_id: str) -> RetrievalHit | None:
        for chunk in self._chunks:
            if chunk["id"] == citation_id:
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
                    topic_tags=chunk["topic_tags"],
                )
        return None
