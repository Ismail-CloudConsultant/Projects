import re

from rag_tool.ingestion.hashing import hash_text
from rag_tool.models import Chunk, LoadedDocument

# Update this list to add or remove KPIs that act as section boundaries.
KPI_LIST = [
    "The Price-to-Earnings (P/E) Ratio",
    "The Forward P/E Ratio",
    "The PEG Ratio (Price/Earnings to Growth)",
    "Price-to-Book (P/B) Ratio",
    "Price-to-Sales (P/S) Ratio",
    "EV/EBITDA",
    "Dividend Yield",
    "Absolute Return",
    "CAGR (Compound Annual Growth Rate)",
    "Alpha",
    "Total Return"
    
]


def _build_heading_pattern(kpis: list[str]) -> re.Pattern:
    escaped = "|".join(re.escape(kpi) for kpi in kpis)
    return re.compile(rf"(\d+\.\s+(?:{escaped})[^\n]*)", re.IGNORECASE)


_HEADING_PATTERN = _build_heading_pattern(KPI_LIST)


def chunk_document(document: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    text = _normalize_whitespace(document.text)
    if not text:
        return []

    sections = _split_by_headings(text)
    if not sections:
        sections = [{"title": "", "content": text}]

    doc_key = hash_text(str(document.path))[:12]
    chunks: list[Chunk] = []

    for index, section in enumerate(sections):
        title = section["title"]
        content = section["content"]
        chunk_text = f"{title}\n\n{content}".strip() if title else content.strip()
        if not chunk_text:
            continue
        content_hash = hash_text(chunk_text)
        chunk_id = f"{doc_key}-{index:05d}-{content_hash[:12]}"
        chunks.append(
            Chunk(
                id=chunk_id,
                document_path=str(document.path),
                text=chunk_text,
                content_hash=content_hash,
                index=index,
                metadata={**document.metadata, "chunk_index": index, "title": title},
            )
        )

    return chunks


def _split_by_headings(text: str) -> list[dict[str, str]]:
    splits = _HEADING_PATTERN.split(text)
    chunks = []
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        chunks.append({"title": title, "content": content})
    return chunks


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines()]
    return "\n".join(line for line in lines if line)
