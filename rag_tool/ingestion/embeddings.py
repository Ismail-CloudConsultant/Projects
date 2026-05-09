import logging
from collections.abc import Iterable

from rag_tool.config import Settings
from rag_tool.utils import retry_with_backoff

logger = logging.getLogger(__name__)


class VertexEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self.settings = settings
        self.client = genai.Client(
            vertexai=False,
            api_key=settings.google_api_key,
        )

    def embed_texts(self, texts: Iterable[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        from google.genai import types

        contents = list(texts)
        if not contents:
            return []

        batch_size = self.settings.embedding_batch_size
        batches = [contents[i : i + batch_size] for i in range(0, len(contents), batch_size)]
        logger.info("Embedding %d texts in %d batch(es) (batch_size=%d).", len(contents), len(batches), batch_size)

        results: list[list[float]] = []
        for batch_index, batch in enumerate(batches, start=1):
            logger.debug("Embedding batch %d/%d (%d texts).", batch_index, len(batches), len(batch))

            def _call(b: list[str] = batch) -> list[list[float]]:
                response = self.client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.settings.embedding_dimensions,
                    ),
                )
                return [list(embedding.values) for embedding in response.embeddings]

            results.extend(retry_with_backoff(_call, max_retries=self.settings.api_max_retries))

        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
