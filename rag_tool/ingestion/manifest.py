from datetime import datetime, timezone
from typing import Any

from rag_tool.gcs_store import GCSStore
from rag_tool.models import Chunk


def load_manifest(store: GCSStore) -> dict[str, Any]:
    return store.read_json("manifest.json") or {"version": 1, "documents": {}}


def save_manifest(store: GCSStore, manifest: dict[str, Any]) -> None:
    store.write_json("manifest.json", manifest)


def has_unchanged_document(manifest: dict[str, Any], source: str, file_hash: str) -> bool:
    entry = manifest.get("documents", {}).get(source)
    return bool(entry and entry.get("file_hash") == file_hash)


def upsert_document(
    manifest: dict[str, Any],
    *,
    source: str,
    file_hash: str,
    text_hash: str,
    chunks: list[Chunk],
) -> None:
    manifest.setdefault("documents", {})[source] = {
        "file_hash": file_hash,
        "text_hash": text_hash,
        "chunk_ids": [chunk.id for chunk in chunks],
        "chunk_hashes": [chunk.content_hash for chunk in chunks],
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def load_chunks(store: GCSStore) -> dict[str, dict[str, Any]]:
    return store.read_json("chunks.json") or {}


def save_chunks(store: GCSStore, chunks: dict[str, dict[str, Any]]) -> None:
    store.write_json("chunks.json", chunks)
