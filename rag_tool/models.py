from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_path: str
    text: str
    content_hash: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

