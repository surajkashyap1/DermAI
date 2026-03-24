from __future__ import annotations

import argparse
import json
from pathlib import Path

from pubmed import PubMedFetchResult, save_normalized_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a raw PubMed dataset into registry-ready JSON source files.")
    parser.add_argument("--dataset-id", required=True, help="Raw dataset identifier created by fetch_pubmed.py.")
    parser.add_argument("--topic-tags", default="", help="Comma-separated topic tags to apply.")
    parser.add_argument("--disease-tags", default="", help="Comma-separated disease tags to apply.")
    return parser.parse_args()


def normalize_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "services" / "ingestion" / "data" / "raw" / "pubmed" / f"{args.dataset_id}.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    dataset = PubMedFetchResult(
        dataset_id=payload["dataset_id"],
        query=payload["query"],
        count=int(payload["count"]),
        ids=payload["ids"],
        articles=payload["articles"],
    )
    output_dir = project_root / "services" / "ingestion" / "data" / "normalized" / "pubmed" / args.dataset_id
    written = save_normalized_articles(
        output_dir=output_dir,
        dataset=dataset,
        topic_tags=normalize_list(args.topic_tags),
        disease_tags=normalize_list(args.disease_tags),
    )
    print(f"Wrote {len(written)} normalized PubMed source files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
