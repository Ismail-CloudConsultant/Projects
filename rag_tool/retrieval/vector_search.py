import logging
from typing import Any

from rag_tool.config import Settings
from rag_tool.gcs_store import GCSStore
from rag_tool.ingestion.embeddings import VertexEmbeddingClient
from rag_tool.ingestion.manifest import load_chunks
from rag_tool.ingestion.sparse import SparseVectorizer
from rag_tool.models import RetrievedChunk
from rag_tool.utils import retry_with_backoff

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_client = VertexEmbeddingClient(settings)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        logger.info("Retrieving top-%d results.", top_k)
        store = GCSStore.from_settings(self.settings)
        dense_embedding = self.embedding_client.embed_query(query)
        sparse_embedding = SparseVectorizer.load(store).transform(query)
        neighbors = self._query_vector_search(dense_embedding, sparse_embedding, top_k)
        return self._hydrate_neighbors(neighbors, store)

    def _query_vector_search(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict[str, list[float] | list[int]],
        top_k: int,
    ) -> list[Any]:
        from google.cloud import aiplatform
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery

        aiplatform.init(
            project=self.settings.google_cloud_project,
            location=self.settings.google_cloud_location,
        )
        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.settings.vector_index_endpoint_id
        )
        hybrid_query = HybridQuery(
            dense_embedding=dense_embedding,
            sparse_embedding_dimensions=sparse_embedding["dimensions"],
            sparse_embedding_values=sparse_embedding["values"],
            rrf_ranking_alpha=self.settings.rrf_ranking_alpha,
        )

        def _call() -> list[Any]:
            result = endpoint.find_neighbors(
                deployed_index_id=self.settings.deployed_index_id,
                queries=[hybrid_query],
                num_neighbors=top_k,
            )
            return result[0] if result else []

        return retry_with_backoff(_call, max_retries=self.settings.api_max_retries)

    def _hydrate_neighbors(self, neighbors: list[Any], store: GCSStore) -> list[RetrievedChunk]:
        chunks = load_chunks(store)
        hydrated: list[RetrievedChunk] = []
        for neighbor in neighbors:
            chunk_id = getattr(neighbor, "id", None) or getattr(neighbor, "datapoint_id", None)
            if not chunk_id:
                continue
            if chunk_id not in chunks:
                logger.warning(
                    "Chunk ID %r returned by Vector Search is not in GCS — skipping.",
                    chunk_id,
                )
                continue
            chunk = chunks[chunk_id]
            hydrated.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=chunk["text"],
                    score=getattr(neighbor, "distance", None),
                    metadata=chunk.get("metadata", {}),
                )
            )
        logger.info("Retrieved %d chunks.", len(hydrated))
        return hydrated
