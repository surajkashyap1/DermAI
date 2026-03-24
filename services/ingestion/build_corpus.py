from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from source_registry import SourceRegistryEntry, load_source_registry


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
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

SCALAR_FIELDS = {
    "source": "source",
    "sourcetype": "source_type",
    "authority": "authority_level",
    "audience": "audience",
    "year": "year",
    "href": "href",
    "pmid": "pmid",
    "authors": "authors",
    "doctype": "doc_type",
}

LIST_FIELDS = {
    "tags": "topic_tags",
    "diseasetags": "disease_tags",
}


@dataclass
class DocumentRecord:
    id: str
    title: str
    source: str
    source_type: str
    authority_level: str
    audience: str
    year: int
    href: str
    pmid: str | None
    authors: list[str]
    doc_type: str
    topic_tags: list[str]
    disease_tags: list[str]
    sections: list[str]


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    title: str
    source: str
    source_type: str
    authority_level: str
    audience: str
    section: str
    section_path: list[str]
    topic_tags: list[str]
    disease_tags: list[str]
    year: int
    href: str
    pmid: str | None
    authors: list[str]
    doc_type: str
    text: str
    snippet: str
    token_counts: dict[str, int]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def split_paragraphs(text: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    buffer = ""

    for block in blocks:
        candidate = f"{buffer}\n\n{block}".strip() if buffer else block
        if len(candidate) <= 900:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
        buffer = block

    if buffer:
        chunks.append(buffer)

    return chunks


def default_metadata(path: Path) -> dict[str, Any]:
    return {
        "source": "DermAI Seed Corpus",
        "source_type": "curated_note",
        "authority_level": "internal_seed",
        "audience": "clinician",
        "year": 2026,
        "href": f"https://dermai.local/sources/{path.stem}",
        "pmid": None,
        "authors": [],
        "doc_type": path.suffix.lstrip(".") or "markdown",
        "topic_tags": [],
        "disease_tags": [],
    }


def normalize_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_registry_metadata(metadata: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**metadata}
    for key, value in overrides.items():
        if key in {"topic_tags", "disease_tags", "authors"}:
            merged[key] = list(value)
        elif key == "year":
            merged[key] = int(value)
        else:
            merged[key] = value
    return merged


def parse_structured_text(path: Path, raw: str, entry: SourceRegistryEntry) -> tuple[DocumentRecord, list[ChunkRecord]]:
    lines = raw.splitlines()
    title = lines[0].replace("#", "").strip() if lines else entry.id.replace("-", " ").title()
    metadata = apply_registry_metadata(default_metadata(path), entry.metadata)
    body_lines: list[str] = []

    for line in lines[1:]:
        if ":" not in line or line.startswith("## "):
            body_lines.append(line)
            continue

        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "")
        value = value.strip()

        if normalized_key in SCALAR_FIELDS:
            field_name = SCALAR_FIELDS[normalized_key]
            metadata[field_name] = int(value) if field_name == "year" else value
            continue

        if normalized_key in LIST_FIELDS:
            metadata[LIST_FIELDS[normalized_key]] = normalize_list(value)
            continue

        body_lines.append(line)

    metadata = apply_registry_metadata(metadata, entry.metadata)

    sections = "\n".join(body_lines).split("\n## ")
    chunks: list[ChunkRecord] = []
    section_titles: list[str] = []
    doc_id = entry.id

    for section_index, section_block in enumerate(sections):
        if not section_block.strip():
            continue

        if section_index == 0:
            section_title = "Overview"
            content = section_block.replace("#", "").strip()
        else:
            first_line, _, remainder = section_block.partition("\n")
            section_title = first_line.strip()
            content = remainder.strip()

        if not content:
            continue

        section_titles.append(section_title)
        section_path = [title, section_title]

        for chunk_index, chunk_text in enumerate(split_paragraphs(content), start=1):
            digest = sha1(f"{doc_id}:{section_title}:{chunk_index}".encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{doc_id}-{digest}"
            tokens = tokenize(chunk_text)
            snippet = chunk_text.replace("\n", " ")

            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    source=metadata["source"],
                    source_type=metadata["source_type"],
                    authority_level=metadata["authority_level"],
                    audience=metadata["audience"],
                    section=section_title,
                    section_path=section_path,
                    topic_tags=metadata["topic_tags"],
                    disease_tags=metadata["disease_tags"],
                    year=metadata["year"],
                    href=metadata["href"],
                    pmid=metadata["pmid"],
                    authors=metadata["authors"],
                    doc_type=metadata["doc_type"],
                    text=chunk_text,
                    snippet=snippet[:320].strip(),
                    token_counts=dict(Counter(tokens)),
                )
            )

    document = DocumentRecord(
        id=doc_id,
        title=title,
        source=metadata["source"],
        source_type=metadata["source_type"],
        authority_level=metadata["authority_level"],
        audience=metadata["audience"],
        year=metadata["year"],
        href=metadata["href"],
        pmid=metadata["pmid"],
        authors=metadata["authors"],
        doc_type=metadata["doc_type"],
        topic_tags=metadata["topic_tags"],
        disease_tags=metadata["disease_tags"],
        sections=section_titles,
    )
    return document, chunks


def parse_pubmed_json(path: Path, entry: SourceRegistryEntry) -> tuple[DocumentRecord, list[ChunkRecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = apply_registry_metadata(
        default_metadata(path),
        {
            "source": payload.get("source", "PubMed Fixture"),
            "source_type": payload.get("source_type", "pubmed_json"),
            "authority_level": payload.get("authority_level", "external_reference"),
            "audience": payload.get("audience", "clinician"),
            "year": int(payload.get("year", 2026)),
            "href": payload.get("href", f"https://dermai.local/sources/{entry.id}"),
            "pmid": payload.get("pmid"),
            "authors": payload.get("authors", []),
            "doc_type": payload.get("doc_type", "json"),
            "topic_tags": payload.get("topic_tags", []),
            "disease_tags": payload.get("disease_tags", []),
        },
    )
    metadata = apply_registry_metadata(metadata, entry.metadata)
    title = payload.get("title", entry.id.replace("-", " ").title())
    sections = payload.get("sections") or [{"title": "Abstract", "text": payload.get("text", "")}]

    synthetic_lines = [f"# {title}"]
    synthetic_lines.extend(
        [
            f"Source: {metadata['source']}",
            f"SourceType: {metadata['source_type']}",
            f"Authority: {metadata['authority_level']}",
            f"Audience: {metadata['audience']}",
            f"Year: {metadata['year']}",
            f"Href: {metadata['href']}",
        ]
    )
    if metadata["pmid"]:
        synthetic_lines.append(f"PMID: {metadata['pmid']}")
    if metadata["authors"]:
        synthetic_lines.append(f"Authors: {', '.join(metadata['authors'])}")
    if metadata["topic_tags"]:
        synthetic_lines.append(f"Tags: {', '.join(metadata['topic_tags'])}")
    if metadata["disease_tags"]:
        synthetic_lines.append(f"DiseaseTags: {', '.join(metadata['disease_tags'])}")
    synthetic_lines.append("")

    for section in sections:
        synthetic_lines.append(f"## {section.get('title', 'Abstract')}")
        synthetic_lines.append(section.get("text", ""))
        synthetic_lines.append("")

    return parse_structured_text(path, "\n".join(synthetic_lines), entry)


def load_source(entry: SourceRegistryEntry, project_root: Path) -> tuple[DocumentRecord, list[ChunkRecord]]:
    path = project_root / entry.path
    if entry.loader == "structured_text":
        return parse_structured_text(path, path.read_text(encoding="utf-8"), entry)
    if entry.loader == "pubmed_json":
        return parse_pubmed_json(path, entry)
    raise ValueError(f"Unsupported loader '{entry.loader}' for source '{entry.id}'")


def build() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "services" / "ingestion" / "data" / "compiled"
    output_dir.mkdir(parents=True, exist_ok=True)

    documents: list[DocumentRecord] = []
    chunks: list[ChunkRecord] = []
    registry_entries = [entry for entry in load_source_registry(project_root) if entry.enabled]
    for entry in registry_entries:
        document, next_chunks = load_source(entry, project_root)
        documents.append(document)
        chunks.extend(next_chunks)

    corpus = {
        "version": "rag-foundation-v2",
        "registryCount": len(registry_entries),
        "documentCount": len(documents),
        "chunkCount": len(chunks),
        "documents": [asdict(document) for document in documents],
        "chunks": [asdict(chunk) for chunk in chunks],
    }

    output_path = output_dir / "corpus.json"
    output_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    return corpus


if __name__ == "__main__":
    result = build()
    print(
        f"Built corpus from {result['registryCount']} registry entries with "
        f"{result['documentCount']} documents and {result['chunkCount']} chunks."
    )
