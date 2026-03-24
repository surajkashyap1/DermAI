from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceRegistryEntry:
    id: str
    path: str
    loader: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def registry_path(project_root: Path) -> Path:
    return project_root / "services" / "ingestion" / "data" / "source-registry.json"


def load_source_registry(project_root: Path) -> list[SourceRegistryEntry]:
    path = registry_path(project_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[SourceRegistryEntry] = []
    for item in payload.get("sources", []):
        entries.append(
            SourceRegistryEntry(
                id=item["id"],
                path=item["path"],
                loader=item["loader"],
                enabled=item.get("enabled", True),
                metadata=item.get("metadata", {}),
            )
        )
    return entries


def write_source_registry(project_root: Path, entries: list[SourceRegistryEntry]) -> None:
    path = registry_path(project_root)
    payload = {
        "version": "v1",
        "sources": [
            {
                "id": entry.id,
                "path": entry.path,
                "loader": entry.loader,
                "enabled": entry.enabled,
                "metadata": entry.metadata,
            }
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
