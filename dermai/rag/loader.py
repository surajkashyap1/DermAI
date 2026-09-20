"""Load curated dermatology sources into LangChain documents.

Supports the three source formats that live in ``data/sources``:

* Markdown clinical notes (``*.md``)
* Structured PubMed-style records (``*.json``)
* The illustrated dermatology textbook (``*.pdf``) used as the knowledge base

Everything is chunked with a recursive splitter so retrieval works over
paragraph-sized passages.
"""

from __future__ import annotations

import json
from pathlib import Path

from dermai.config import CHUNK_OVERLAP, CHUNK_SIZE, SOURCES_DIR


def _load_markdown(path: Path) -> list[tuple[str, dict]]:
    text = path.read_text(encoding="utf-8")
    return [(text, {"source": path.name, "type": "markdown"})]


def _load_json(path: Path) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]
    out: list[tuple[str, dict]] = []
    for record in records:
        if isinstance(record, dict):
            # Flatten string-ish fields into a readable passage.
            parts = [f"{k}: {v}" for k, v in record.items() if isinstance(v, (str, int, float))]
            out.append(("\n".join(parts), {"source": path.name, "type": "json"}))
        else:
            out.append((str(record), {"source": path.name, "type": "json"}))
    return out


def _load_pdf(path: Path) -> list[tuple[str, dict]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[str, dict]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": path.name, "type": "pdf", "page": page_number}))
    return pages


def load_documents(sources_dir: Path = SOURCES_DIR) -> list:
    """Return chunked LangChain ``Document`` objects from all curated sources."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sources_dir = Path(sources_dir)
    raw: list[tuple[str, dict]] = []
    for path in sorted(sources_dir.iterdir()):
        if path.suffix == ".md":
            raw.extend(_load_markdown(path))
        elif path.suffix == ".json":
            raw.extend(_load_json(path))
        elif path.suffix == ".pdf":
            raw.extend(_load_pdf(path))

    if not raw:
        raise FileNotFoundError(f"No curated sources found in {sources_dir}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = [Document(page_content=text, metadata=meta) for text, meta in raw]
    return splitter.split_documents(documents)
