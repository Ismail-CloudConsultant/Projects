from hashlib import sha256
from pathlib import Path


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def hash_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip().encode("utf-8")
    return hash_bytes(normalized)


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

