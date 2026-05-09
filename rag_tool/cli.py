import logging
from pathlib import Path

import typer
from rich.console import Console

from rag_tool.config import get_settings
from rag_tool.generation.generator import RagGenerator
from rag_tool.ingestion.pipeline import IngestionPipeline
from rag_tool.provisioning.vector_index import VectorSearchProvisioner
from rag_tool.retrieval.vector_search import HybridRetriever
from rag_tool.utils import configure_logging

app = typer.Typer(help="CLI for a Vertex AI Vector Search RAG tool.")
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging.")) -> None:
    configure_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def ingest(
    documents_dir: Path = typer.Option(Path("documents"), help="Directory containing PDF/TXT/MD files."),
    no_embed: bool = typer.Option(False, help="Skip Vertex embedding calls and write zero vectors."),
) -> None:
    """Parse documents, deduplicate, chunk, embed, and write index JSONL."""
    settings = get_settings()
    result = IngestionPipeline(settings).run(documents_dir, embed=not no_embed)
    console.print(
        f"Seen={result.documents_seen} changed={result.documents_changed} "
        f"chunks={result.chunks_written} payload={result.index_payload_path}"
    )


@app.command("upload-index")
def upload_index(
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace all index vectors (clean re-index)."),
) -> None:
    """Upload index payload to GCS and trigger a batch update on the existing Vertex AI index."""
    count = VectorSearchProvisioner(get_settings()).upload(overwrite=overwrite)
    console.print(f"Uploaded {count} datapoints.")


@app.command("provision-index")
def provision_index() -> None:
    """Create and deploy a brand-new Vertex AI Vector Search index via GCS."""
    result = VectorSearchProvisioner(get_settings()).provision()
    for key, value in result.items():
        console.print(f"{key}: {value}")


@app.command()
def retrieve(query: str, top_k: int | None = typer.Option(None, help="Override TOP_K.")) -> None:
    """Run hybrid retrieval and print matched chunks."""
    chunks = HybridRetriever(get_settings()).retrieve(query, top_k=top_k)
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", chunk.id)
        console.print(f"\n[{index}] score={chunk.score} source={source}\n{chunk.text}")


@app.command()
def ask(question: str) -> None:
    """Run retrieval and generate an answer with Gemini."""
    console.print(RagGenerator(get_settings()).answer(question))


if __name__ == "__main__":
    app()
