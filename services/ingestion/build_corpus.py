from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path


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


@dataclass
class ChunkRecord:
    id: str
    source_id: str
    title: str
    source: str
    section: str
    topic_tags: list[str]
    year: int
    href: str
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
        if len(candidate) <= 700:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
        buffer = block

    if buffer:
        chunks.append(buffer)

    return chunks


def parse_source(path: Path) -> list[ChunkRecord]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title = lines[0].replace("#", "").strip()
    metadata = {
        "source": "DermAI Seed Corpus",
        "year": 2026,
        "href": f"https://dermai.local/sources/{path.stem}",
        "tags": [],
    }
    body_lines: list[str] = []

    for line in lines[1:]:
        if line.startswith("Source:"):
            metadata["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("Year:"):
            metadata["year"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Href:"):
            metadata["href"] = line.split(":", 1)[1].strip()
        elif line.startswith("Tags:"):
            metadata["tags"] = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
        else:
            body_lines.append(line)

    sections = "\n".join(body_lines).split("\n## ")
    chunks: list[ChunkRecord] = []
    source_id = path.stem

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

        for chunk_index, chunk_text in enumerate(split_paragraphs(content), start=1):
            digest = sha1(f"{source_id}:{section_title}:{chunk_index}".encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{source_id}-{digest}"
            tokens = tokenize(chunk_text)
            snippet = chunk_text.replace("\n", " ")

            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    source_id=source_id,
                    title=title,
                    source=metadata["source"],
                    section=section_title,
                    topic_tags=metadata["tags"],
                    year=metadata["year"],
                    href=metadata["href"],
                    text=chunk_text,
                    snippet=snippet[:260].strip(),
                    token_counts=dict(Counter(tokens)),
                )
            )

    return chunks


def build() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    source_dir = project_root / "services" / "ingestion" / "data" / "sources"
    output_dir = project_root / "services" / "ingestion" / "data" / "compiled"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[ChunkRecord] = []
    for path in sorted(source_dir.glob("*.md")):
        chunks.extend(parse_source(path))

    corpus = {
        "version": "phase-2-seed",
        "chunkCount": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }

    output_path = output_dir / "corpus.json"
    output_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    return corpus


if __name__ == "__main__":
    result = build()
    print(f"Built corpus with {result['chunkCount']} chunks.")
