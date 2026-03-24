from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.retrieval import RetrievalService  # noqa: E402


def load_cases() -> list[dict[str, object]]:
    benchmark_path = REPO_ROOT / "services" / "eval" / "retrieval-benchmark.json"
    return json.loads(benchmark_path.read_text(encoding="utf-8"))


def artifact_path() -> Path:
    path = REPO_ROOT / "services" / "eval" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path / "retrieval-latest.json"


def hit_matches(hit: dict[str, object], case: dict[str, object]) -> bool:
    title = str(hit["title"]).lower()
    disease_tags = {tag.lower() for tag in hit["diseaseTags"]}
    topic_tags = {tag.lower() for tag in hit["topicTags"]}

    expected_title_fragments = [fragment.lower() for fragment in case.get("expected_title_fragments", [])]
    if expected_title_fragments and any(fragment in title for fragment in expected_title_fragments):
        return True

    expected_disease_tags = {tag.lower() for tag in case.get("expected_disease_tags", [])}
    if expected_disease_tags and disease_tags.intersection(expected_disease_tags):
        return True

    expected_topic_tags = {tag.lower() for tag in case.get("expected_topic_tags", [])}
    if expected_topic_tags and topic_tags.intersection(expected_topic_tags):
        return True

    return False


def main() -> int:
    retrieval = RetrievalService()
    cases = load_cases()
    checks: list[dict[str, object]] = []

    for case in cases:
        result = retrieval.debug_search(str(case["query"]), top_k=5)
        top_hit = result["hits"][0] if result["hits"] else None
        top1_pass = bool(top_hit) and hit_matches(
            {
                "title": top_hit.title,
                "diseaseTags": top_hit.disease_tags,
                "topicTags": top_hit.topic_tags,
            },
            case,
        )
        top3_pass = any(
            hit_matches(
                {
                    "title": hit.title,
                    "diseaseTags": hit.disease_tags,
                    "topicTags": hit.topic_tags,
                },
                case,
            )
            for hit in result["hits"][:3]
        )
        checks.append(
            {
                "name": case["name"],
                "query": case["query"],
                "backend": result["backend"],
                "candidateCount": result.get("candidate_count", len(result["hits"])),
                "topHitTitle": top_hit.title if top_hit else None,
                "topHitScore": top_hit.score if top_hit else None,
                "top1Passed": top1_pass,
                "top3Passed": top3_pass,
            }
        )

    top1_passed = sum(1 for check in checks if check["top1Passed"])
    top3_passed = sum(1 for check in checks if check["top3Passed"])
    report = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "checks": checks,
        "summary": {
            "benchmarkCount": len(checks),
            "top1Passed": top1_passed,
            "top3Passed": top3_passed,
            "failed": len(checks) - top3_passed,
            "allPassed": top3_passed == len(checks),
        },
    }

    output = artifact_path()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved retrieval eval artifact to {output}")
    return 0 if report["summary"]["allPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
