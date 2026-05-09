import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_tool.config import Settings
from rag_tool.gcs_store import GCSStore
from rag_tool.ingestion.chunking import chunk_document
from rag_tool.ingestion.embeddings import VertexEmbeddingClient
from rag_tool.ingestion.hashing import hash_file, hash_text
from rag_tool.ingestion.loaders import iter_document_paths, load_document
from rag_tool.ingestion.manifest import (
    has_unchanged_document,
    load_chunks,
    load_manifest,
    save_chunks,
    save_manifest,
    upsert_document,
)
from rag_tool.ingestion.sparse import SparseVectorizer
from rag_tool.models import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    documents_seen: int
    documents_changed: int
    chunks_written: int
    index_payload_path: Path


class IngestionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, documents_dir: Path | None = None, *, embed: bool = True) -> IngestionResult:
        documents_dir = documents_dir or self.settings.documents_dir
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

        store = GCSStore.from_settings(self.settings)
        manifest = load_manifest(store)
        chunk_store = load_chunks(store)
        document_paths = iter_document_paths(documents_dir)
        current_sources = {str(p) for p in document_paths}

        self._prune_deleted(manifest, chunk_store, current_sources)

        archive_dir = documents_dir / "archive"
        changed_documents = 0
        for path in document_paths:
            source = str(path)
            file_hash = hash_file(path)
            if has_unchanged_document(manifest, source, file_hash):
                logger.debug("Skipping unchanged document: %s", source)
            else:
                logger.info("Processing changed document: %s", source)
                document = load_document(path)
                text_hash = hash_text(document.text)
                chunks = chunk_document(document, self.settings.chunk_size, self.settings.chunk_overlap)
                self._remove_old_chunks(manifest, chunk_store, source)
                for chunk in chunks:
                    chunk_store[chunk.id] = _chunk_to_json(chunk)
                upsert_document(
                    manifest,
                    source=source,
                    file_hash=file_hash,
                    text_hash=text_hash,
                    chunks=chunks,
                )
                changed_documents += 1

            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / path.name
            if dest.exists():
                dest = archive_dir / f"{path.stem}_{file_hash[:8]}{path.suffix}"
            path.rename(dest)
            logger.info("Archived: %s → %s", path.name, dest)

        active_chunks = list(chunk_store.values())
        if active_chunks:
            logger.info("Fitting sparse vectorizer on %d chunks.", len(active_chunks))
            vectorizer = SparseVectorizer.fit([chunk["text"] for chunk in active_chunks])
            vectorizer.save(store)
            self._write_index_payload(active_chunks, vectorizer, embed=embed)
        else:
            logger.warning("No active chunks — writing empty index payload.")
            self._write_empty_index_payload()

        save_manifest(store, manifest)
        save_chunks(store, chunk_store)
        logger.info(
            "Ingestion complete: seen=%d changed=%d chunks=%d",
            len(document_paths),
            changed_documents,
            len(active_chunks),
        )

        return IngestionResult(
            documents_seen=len(document_paths),
            documents_changed=changed_documents,
            chunks_written=len(active_chunks),
            index_payload_path=self.settings.index_payload_path,
        )

    def _prune_deleted(
        self,
        manifest: dict[str, Any],
        chunk_store: dict[str, dict[str, Any]],
        current_sources: set[str],
    ) -> None:
        deleted = [src for src in manifest.get("documents", {}) if src not in current_sources]
        for source in deleted:
            logger.info("Pruning deleted document from manifest: %s", source)
            self._remove_old_chunks(manifest, chunk_store, source)
            del manifest["documents"][source]

    @staticmethod
    def _remove_old_chunks(
        manifest: dict[str, Any],
        chunk_store: dict[str, dict[str, Any]],
        source: str,
    ) -> None:
        for chunk_id in manifest.get("documents", {}).get(source, {}).get("chunk_ids", []):
            chunk_store.pop(chunk_id, None)

    def _write_index_payload(
        self,
        chunks: list[dict[str, Any]],
        vectorizer: SparseVectorizer,
        *,
        embed: bool,
    ) -> None:
        self.settings.index_payload_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self._embed_chunks(chunks) if embed else [[0.0] * self.settings.embedding_dimensions for _ in chunks]
        with self.settings.index_payload_path.open("w", encoding="utf-8") as file:
            for chunk, dense_embedding in zip(chunks, embeddings, strict=True):
                item = {
                    "id": chunk["id"],
                    "embedding": dense_embedding,
                    "sparse_embedding": vectorizer.transform(chunk["text"]),
                    "restricts": [{"namespace": "source", "allow": [chunk["document_path"]]}],
                }
                file.write(json.dumps(item) + "\n")
        logger.info("Wrote index payload to %s.", self.settings.index_payload_path)

    def _write_empty_index_payload(self) -> None:
        self.settings.index_payload_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.index_payload_path.write_text("", encoding="utf-8")

    def _embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        client = VertexEmbeddingClient(self.settings)
        return client.embed_texts([chunk["text"] for chunk in chunks])


def _chunk_to_json(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_path": chunk.document_path,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "index": chunk.index,
        "metadata": chunk.metadata,
    }
