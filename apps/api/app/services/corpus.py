from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def compiled_corpus_path() -> Path:
    return project_root() / "services" / "ingestion" / "data" / "compiled" / "corpus.json"


def load_compiled_corpus() -> dict[str, Any]:
    path = compiled_corpus_path()
    if not path.exists():
        raise FileNotFoundError(
            "Compiled corpus not found. Run `python services/ingestion/build_corpus.py` first."
        )

    return json.loads(path.read_text(encoding="utf-8"))
