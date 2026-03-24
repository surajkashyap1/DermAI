from __future__ import annotations

import argparse
from pathlib import Path

from source_registry import SourceRegistryEntry, load_source_registry, write_source_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add normalized PubMed JSON files to the DermAI source registry.")
    parser.add_argument("--dataset-id", required=True, help="Normalized dataset directory name under data/normalized/pubmed.")
    parser.add_argument("--topic-tags", default="", help="Comma-separated topic tags to apply to new entries.")
    parser.add_argument("--disease-tags", default="", help="Comma-separated disease tags to apply to new entries.")
    parser.add_argument("--audience", default="clinician", help="Audience metadata for added entries.")
    return parser.parse_args()


def normalize_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    normalized_dir = project_root / "services" / "ingestion" / "data" / "normalized" / "pubmed" / args.dataset_id
    if not normalized_dir.exists():
        raise FileNotFoundError(f"Normalized dataset directory not found: {normalized_dir}")

    normalized_files = sorted(normalized_dir.glob("*.json"))
    entries = load_source_registry(project_root)
    existing_ids = {entry.id for entry in entries}

    added = 0
    for path in normalized_files:
        entry_id = path.stem
        if entry_id in existing_ids:
            continue

        relative_path = path.relative_to(project_root).as_posix()
        entries.append(
            SourceRegistryEntry(
                id=entry_id,
                path=relative_path,
                loader="pubmed_json",
                enabled=True,
                metadata={
                    "topic_tags": normalize_list(args.topic_tags),
                    "disease_tags": normalize_list(args.disease_tags),
                    "audience": args.audience,
                },
            )
        )
        existing_ids.add(entry_id)
        added += 1

    write_source_registry(project_root, entries)
    print(
        f"Scanned {len(normalized_files)} normalized files and added {added} "
        f"PubMed entries to the source registry from dataset '{args.dataset_id}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
