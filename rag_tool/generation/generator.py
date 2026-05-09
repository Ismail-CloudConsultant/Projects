import logging

from rag_tool.config import Settings
from rag_tool.models import RetrievedChunk
from rag_tool.retrieval.vector_search import HybridRetriever
from rag_tool.utils import retry_with_backoff

logger = logging.getLogger(__name__)


class RagGenerator:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self.client = genai.Client(
            vertexai=False,
            api_key=settings.google_api_key,
        )

    def answer(self, question: str) -> str:
        chunks = self.retriever.retrieve(question)
        prompt = build_prompt(question, chunks)
        logger.info("Generating answer from %d context chunk(s).", len(chunks))

        def _call() -> str:
            response = self.client.models.generate_content(
                model=self.settings.generation_model,
                contents=prompt,
            )
            return response.text or ""

        return retry_with_backoff(_call, max_retries=self.settings.api_max_retries)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{index}] source={chunk.metadata.get('source', chunk.id)}\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return (
        "You are a financial education assistant. Follow these rules strictly:\n\n"
        "1. SCOPE: Only answer questions related to finance, investing, stocks, markets, "
        "financial metrics, or economic concepts. If the question is unrelated to finance "
        "(e.g. cooking, sports, general trivia), respond with: "
        "'I only answer finance and investment-related questions.'\n\n"
        "2. USE CORPUS FIRST: If the answer is present in the context below, answer using "
        "that information and cite the source.\n\n"
        "3. FALLBACK TO KNOWLEDGE: If the answer is NOT in the context but the question "
        "is finance-related, answer using your own knowledge. Start your response with: "
        "'(Not in knowledge base) ' and then explain the concept clearly in simple, "
        "everyday language that a beginner investor would understand. Always include:\n"
        "   - A plain-English definition\n"
        "   - A real-world analogy or example with numbers\n"
        "   - Why it matters for an investor\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
