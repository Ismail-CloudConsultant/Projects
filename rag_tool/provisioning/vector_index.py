import logging
from pathlib import Path

from rag_tool.config import Settings
from rag_tool.utils import retry_with_backoff

logger = logging.getLogger(__name__)


class VectorSearchProvisioner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def upload(self, payload_path: Path | None = None, *, overwrite: bool = False) -> int:
        """Upload items.jsonl to GCS then trigger a batch index update."""
        from google.cloud import aiplatform

        payload_path = payload_path or self.settings.index_payload_path
        if not payload_path.exists():
            raise FileNotFoundError(f"Index payload does not exist: {payload_path}")
        if not self.settings.vector_index_id:
            raise ValueError("VECTOR_INDEX_ID is required for upload")
        if not self.settings.gcs_bucket_uri:
            raise ValueError("GCS_BUCKET_URI is required for batch upload")

        total = sum(1 for line in payload_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if total == 0:
            logger.warning("No items in payload — nothing to upload.")
            return 0

        logger.info("Uploading %d datapoints to GCS%s.", total, " (overwrite=True)" if overwrite else "")
        gcs_uri = self._upload_payload(payload_path)

        aiplatform.init(
            project=self.settings.google_cloud_project,
            location=self.settings.google_cloud_location,
        )
        index = aiplatform.MatchingEngineIndex(self.settings.vector_index_id)

        def _call() -> None:
            index.update_embeddings(contents_delta_uri=gcs_uri, is_complete_overwrite=overwrite)
            logger.info("Batch update triggered for index %s.", self.settings.vector_index_id)

        retry_with_backoff(_call, max_retries=self.settings.api_max_retries)
        logger.info("Batch update submitted. Index will be ready in a few minutes.")
        return total

    def provision(self, payload_path: Path | None = None) -> dict[str, str]:
        from google.cloud import aiplatform

        payload_path = payload_path or self.settings.index_payload_path
        if not payload_path.exists():
            raise FileNotFoundError(f"Index payload does not exist: {payload_path}")
        if not self.settings.gcs_bucket_uri:
            raise ValueError("GCS_BUCKET_URI is required for provisioning")

        aiplatform.init(
            project=self.settings.google_cloud_project,
            location=self.settings.google_cloud_location,
        )
        contents_delta_uri = self._upload_payload(payload_path)
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name="rag-tool-hybrid-index",
            contents_delta_uri=contents_delta_uri,
            dimensions=self.settings.embedding_dimensions,
            approximate_neighbors_count=max(self.settings.top_k, 10),
        )
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name="rag-tool-index-endpoint",
            public_endpoint_enabled=True,
        )
        endpoint.deploy_index(index=index, deployed_index_id=self.settings.deployed_index_id)
        return {
            "index_id": index.resource_name,
            "index_endpoint_id": endpoint.resource_name,
            "deployed_index_id": self.settings.deployed_index_id,
            "contents_delta_uri": contents_delta_uri,
        }

    def _upload_payload(self, payload_path: Path) -> str:
        from google.cloud import storage

        bucket_name, prefix = _parse_gcs_uri(self.settings.gcs_bucket_uri)
        # Vertex AI batch update scans a directory prefix; files must use .json extension.
        folder = f"{prefix.rstrip('/')}/index" if prefix else "index"
        destination = f"{folder}/items.json"
        client = storage.Client(project=self.settings.google_cloud_project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination)
        blob.upload_from_filename(str(payload_path))
        return f"gs://{bucket_name}/{folder}"


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("GCS_BUCKET_URI must start with gs://")
    without_scheme = uri.removeprefix("gs://")
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError("GCS_BUCKET_URI must include a bucket name")
    return bucket, prefix

