from __future__ import annotations

import asyncio
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.chat_runtime import chat_runtime  # noqa: E402
from app.services.vision import vision_service  # noqa: E402


def artifact_path() -> Path:
    path = REPO_ROOT / "services" / "eval" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path / "phase6-latest.json"


async def run_eval() -> dict[str, object]:
    report: dict[str, object] = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "checks": [],
        "summary": {},
    }
    checks: list[dict[str, object]] = []

    chat_cases = [
        {
            "name": "greeting",
            "message": "hello",
            "mode": "chat",
            "expect_citations": False,
        },
        {
            "name": "grounded_definition",
            "message": "What is melanoma?",
            "mode": "chat",
            "expect_citations": True,
        },
        {
            "name": "off_topic_redirect",
            "message": "What is the weather in Delhi?",
            "mode": "chat",
            "expect_citations": False,
        },
        {
            "name": "emergency_escalation",
            "message": "I have a rapidly spreading rash and trouble breathing",
            "mode": "chat",
            "expect_citations": False,
        },
    ]

    for case in chat_cases:
        response = await chat_runtime.answer(None, case["message"], case["mode"])
        passed = bool(response.answer) and (
            len(response.citations) > 0 if case["expect_citations"] else True
        )
        checks.append(
            {
                "name": case["name"],
                "passed": passed,
                "confidence": response.confidence,
                "citationCount": len(response.citations),
                "followUps": response.followUps,
            }
        )

    image = Image.new("RGB", (144, 144), (126, 92, 84))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    analysis = vision_service.analyze(buffer.getvalue(), "image/png")
    session = chat_runtime.sessions.get_or_create(None)
    chat_runtime.sessions.attach_image_analysis(session.session_id, analysis)

    multimodal = await chat_runtime.answer(
        session.session_id,
        "What does this image result mean and when should it be reviewed?",
        "image_follow_up",
    )
    checks.append(
        {
            "name": "multimodal_follow_up",
            "passed": bool(multimodal.answer) and len(multimodal.citations) > 0,
            "confidence": multimodal.confidence,
            "citationCount": len(multimodal.citations),
            "imagePredictedClass": analysis.predictedClass,
            "imageConfidenceBand": analysis.confidenceBand,
        }
    )

    passed_count = sum(1 for check in checks if check["passed"])
    report["checks"] = checks
    report["summary"] = {
        "passed": passed_count,
        "failed": len(checks) - passed_count,
        "allPassed": passed_count == len(checks),
    }
    return report


def main() -> int:
    report = asyncio.run(run_eval())
    output = artifact_path()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved eval artifact to {output}")
    return 0 if report["summary"]["allPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
