"""Document ingestion pipeline."""

from rag_tool.ingestion.embeddings import VertexEmbeddingClient
from rag_tool.ingestion.pipeline import IngestionPipeline, IngestionResult
from rag_tool.ingestion.sparse import SparseVectorizer

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "SparseVectorizer",
    "VertexEmbeddingClient",
]
