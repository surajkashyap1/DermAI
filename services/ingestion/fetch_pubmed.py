from __future__ import annotations

import argparse
from pathlib import Path

from pubmed import fetch_dataset, save_raw_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PubMed search results into a local raw dataset file.")
    parser.add_argument("--dataset-id", required=True, help="Stable local dataset identifier.")
    parser.add_argument("--query", required=True, help="PubMed search query.")
    parser.add_argument("--retmax", type=int, default=20, help="Maximum number of records to fetch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "services" / "ingestion" / "data" / "raw" / "pubmed" / f"{args.dataset_id}.json"
    dataset = fetch_dataset(query=args.query, dataset_id=args.dataset_id, retmax=args.retmax)
    save_raw_dataset(output_path, dataset)
    print(f"Saved PubMed raw dataset '{args.dataset_id}' with {len(dataset.articles)} fetched articles to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
