import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]

        if end < len(text):
            boundary = chunk.rfind(". ")
            if boundary > chunk_size * 0.6:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks
