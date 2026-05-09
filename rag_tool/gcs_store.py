import io
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GCSStore:
    """Read/write JSON and binary blobs to a GCS bucket prefix."""

    def __init__(self, bucket_name: str, prefix: str, project: str) -> None:
        from google.cloud import storage

        self._bucket = storage.Client(project=project).bucket(bucket_name)
        self._prefix = prefix.rstrip("/")

    def _blob(self, name: str):
        key = f"{self._prefix}/{name}" if self._prefix else name
        return self._bucket.blob(key)

    def read_json(self, name: str) -> dict[str, Any] | None:
        blob = self._blob(name)
        if not blob.exists():
            return None
        logger.debug("Reading %s from GCS.", name)
        return json.loads(blob.download_as_text(encoding="utf-8"))

    def write_json(self, name: str, data: dict[str, Any]) -> None:
        logger.debug("Writing %s to GCS.", name)
        self._blob(name).upload_from_string(
            json.dumps(data, indent=2, sort_keys=True),
            content_type="application/json",
        )

    def read_bytes(self, name: str) -> bytes | None:
        blob = self._blob(name)
        if not blob.exists():
            return None
        logger.debug("Reading %s (bytes) from GCS.", name)
        return blob.download_as_bytes()

    def write_bytes(self, name: str, data: bytes) -> None:
        logger.debug("Writing %s (bytes) to GCS.", name)
        self._blob(name).upload_from_string(data, content_type="application/octet-stream")

    @classmethod
    def from_settings(cls, settings: "Any") -> "GCSStore":
        from rag_tool.provisioning.vector_index import _parse_gcs_uri

        bucket_name, _ = _parse_gcs_uri(settings.gcs_bucket_uri)
        return cls(bucket_name=bucket_name, prefix="rag", project=settings.google_cloud_project)
