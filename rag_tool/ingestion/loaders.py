from pathlib import Path

from rag_tool.models import LoadedDocument


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def iter_document_paths(documents_dir: Path) -> list[Path]:
    if not documents_dir.exists():
        return []
    archive_dir = documents_dir / "archive"
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.is_relative_to(archive_dir)
    )


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")

    return LoadedDocument(
        path=path,
        text=text,
        metadata={"source": str(path), "extension": suffix},
    )


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[page {page_number}]\n{page_text}")
    return "\n\n".join(pages)

