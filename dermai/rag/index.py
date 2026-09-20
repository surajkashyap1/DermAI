"""Build and load the FAISS vector index over curated dermatology sources."""

from __future__ import annotations

from pathlib import Path

from dermai.config import FAISS_INDEX_DIR
from dermai.rag.embeddings import get_embeddings
from dermai.rag.loader import load_documents


def build_index(index_dir: Path = FAISS_INDEX_DIR, force: bool = False):
    """Create (or reuse) a FAISS index and persist it to ``index_dir``."""
    from langchain_community.vectorstores import FAISS

    index_dir = Path(index_dir)
    embeddings = get_embeddings()

    if index_dir.exists() and not force:
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    documents = load_documents()
    store = FAISS.from_documents(documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return store


def load_index(index_dir: Path = FAISS_INDEX_DIR):
    """Load an existing index, building it on first use if necessary."""
    from langchain_community.vectorstores import FAISS

    index_dir = Path(index_dir)
    if not index_dir.exists():
        return build_index(index_dir)
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    store = build_index(force=True)
    print(f"FAISS index built at {FAISS_INDEX_DIR}")
