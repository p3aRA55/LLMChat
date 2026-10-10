from __future__ import annotations

import re


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟،])\s+|[\r\n]+")


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1
    return [chunk for chunk in chunks if chunk]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")
    if not text or not text.strip():
        return []

    units = [
        unit.strip()
        for unit in _SENTENCE_BOUNDARY.split(text.strip())
        if unit.strip()
    ]
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            long_chunks = _split_long_text(unit, chunk_size, overlap)
            chunks.extend(long_chunks[:-1])
            current = long_chunks[-1]
            if not overlap:
                chunks.append(long_chunks[-1])
                current = ""
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) > chunk_size:
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
