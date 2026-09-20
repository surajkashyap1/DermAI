"""End-to-end dermatology RAG: retrieve from FAISS, then generate with Groq."""

from __future__ import annotations

from dataclasses import dataclass

from dermai.config import RETRIEVER_TOP_K
from dermai.rag.index import load_index
from dermai.rag.llm import generate_answer, is_smalltalk, smalltalk_answer


@dataclass
class RAGResult:
    answer: str
    provider: str
    sources: list[dict]


class DermatologyRAG:
    """Lazy-loaded retriever + generator over the curated dermatology corpus."""

    def __init__(self, top_k: int = RETRIEVER_TOP_K):
        self.top_k = top_k
        self._store = None

    def _ensure_store(self):
        if self._store is None:
            self._store = load_index()
        return self._store

    def ask(self, question: str, chat_history: list | None = None) -> RAGResult:
        # Greetings / small talk shouldn't hit retrieval or show sources.
        if is_smalltalk(question):
            answer = smalltalk_answer(question, chat_history=chat_history)
            return RAGResult(answer=answer.text, provider=answer.provider, sources=[])

        store = self._ensure_store()
        documents = store.similarity_search(question, k=self.top_k)
        answer = generate_answer(question, documents, chat_history=chat_history)
        sources = [
            {
                "source": doc.metadata.get("source", "source"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content.strip()[:220],
            }
            for doc in documents
        ]
        return RAGResult(answer=answer.text, provider=answer.provider, sources=sources)
