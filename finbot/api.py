"""FastAPI server for Finbot — wraps the ADK agent with an HTTP API."""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from .agent import root_agent  # noqa: E402

logger = logging.getLogger(__name__)

_APP_NAME = "finbot"

session_service: InMemorySessionService
runner: Runner


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global session_service, runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=_APP_NAME,
        session_service=session_service,
        auto_create_session=True,
    )
    yield


app = FastAPI(title="Finbot API", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    session_id: str
    response: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=request.message)],
    )
    last_response: list[str] = []
    try:
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            if event.is_final_response() and event.content:
                # Overwrite on each final event — only the root agent's last
                # response is the user-facing reply; sub-agent finals are noise.
                last_response = [
                    p.text
                    for p in event.content.parts
                    if getattr(p, "text", None)
                ]
    except Exception as exc:
        logger.exception("chat handler failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(session_id=session_id, response="".join(last_response))


# ---------------------------------------------------------------------------
# RAG ingestion endpoints
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent.parent / ".env"
_DOCUMENTS_DIR = Path(__file__).parent.parent / "rag_tool" / "documents"


class IngestResponse(BaseModel):
    documents_seen: int
    documents_changed: int
    chunks_written: int
    index_payload_path: str


class UploadIndexResponse(BaseModel):
    datapoints_uploaded: int


def _run_ingestion() -> IngestResponse:
    from rag_tool.config import Settings
    from rag_tool.ingestion.pipeline import IngestionPipeline

    settings = Settings()
    result = IngestionPipeline(settings).run(_DOCUMENTS_DIR)
    return IngestResponse(
        documents_seen=result.documents_seen,
        documents_changed=result.documents_changed,
        chunks_written=result.chunks_written,
        index_payload_path=str(result.index_payload_path),
    )


def _run_upload_index(overwrite: bool) -> int:
    from rag_tool.config import Settings
    from rag_tool.provisioning.vector_index import VectorSearchProvisioner

    settings = Settings()
    return VectorSearchProvisioner(settings).upload(overwrite=overwrite)


def _run_rag_ask(question: str) -> str:
    from rag_tool.config import Settings
    from rag_tool.generation.generator import RagGenerator

    return RagGenerator(Settings()).answer(question)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


@app.post("/rag/ask", response_model=AskResponse, summary="Ask a finance question answered by the RAG corpus")
async def rag_ask(request: AskRequest) -> AskResponse:
    """Retrieve relevant chunks and generate an answer. Falls back to internal knowledge for finance topics not in the corpus."""
    try:
        answer = await asyncio.to_thread(_run_rag_ask, request.question)
        return AskResponse(question=request.question, answer=answer)
    except Exception as exc:
        logger.exception("rag ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/upload-document", summary="Upload a document file to the ingestion queue")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Save an uploaded PDF/TXT/MD file to the documents directory for ingestion."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md.")
    _DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DOCUMENTS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse({"saved": str(dest), "size_bytes": len(content)})


@app.post("/rag/ingest", response_model=IngestResponse, summary="Ingest documents and build index payload")
async def ingest_documents() -> IngestResponse:
    """Parse, chunk, embed all documents in the documents directory and write the index JSONL."""
    try:
        return await asyncio.to_thread(_run_ingestion)
    except Exception as exc:
        logger.exception("ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/upload-index", response_model=UploadIndexResponse, summary="Upload index payload to Vertex AI")
async def upload_index(overwrite: bool = False) -> UploadIndexResponse:
    """Push the local index JSONL to GCS and trigger a batch update on the Vertex AI index."""
    try:
        count = await asyncio.to_thread(_run_upload_index, overwrite)
        return UploadIndexResponse(datapoints_uploaded=count)
    except Exception as exc:
        logger.exception("index upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
