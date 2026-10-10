from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.chunking import chunk_text
from rag.embeddings import Embedder
from rag.vector_store import NumpyVectorStore, VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        data_dir: str | Path = "rag/data",
    ) -> None:
        self.embedder = embedder if embedder is not None else Embedder()
        self.store = store if store is not None else NumpyVectorStore()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store.load(self.data_dir)

    def _save(self) -> None:
        self.store.save(self.data_dir)

    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(text)
        vectors = self.embedder.embed_documents(chunks) if chunks else None
        self.store.delete_by_source(source)
        if chunks and vectors is not None:
            metadatas = [
                {"text": chunk, "source": source, "chunk_index": index}
                for index, chunk in enumerate(chunks)
            ]
            self.store.add(vectors, metadatas)
        self._save()
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if file_path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Only .txt and .md files supported")
        return self.ingest_text(
            file_path.read_text(encoding="utf-8"),
            file_path.name,
        )

    def retrieve(
        self, query: str, top_k: int = 4, min_score: float = 0.3
    ) -> list[dict[str, Any]]:
        if len(self.store) == 0:
            return []
        query_vector = self.embedder.embed_query(query)
        return [
            {"text": metadata["text"], "source": metadata["source"], "score": score}
            for metadata, score in self.store.search(query_vector, top_k)
            if score >= min_score
        ]

    def build_context(
        self, results: list[dict[str, Any]], max_chars: int = 3000
    ) -> str:
        if max_chars <= 0:
            return ""
        lines: list[str] = []
        remaining = max_chars
        for index, result in enumerate(results, 1):
            prefix = f"[{index}] "
            suffix = f" (source: {result['source']})"
            available_text = remaining - len(prefix) - len(suffix)
            if available_text <= 0:
                break
            text = str(result["text"])[:available_text]
            line = f"{prefix}{text}{suffix}"
            lines.append(line)
            remaining -= len(line) + 1
            if len(text) < len(str(result["text"])):
                break
        return "\n".join(lines)

    def delete_source(self, source: str) -> int:
        deleted = self.store.delete_by_source(source)
        if deleted:
            self._save()
        return deleted

    def list_sources(self) -> dict[str, int]:
        return self.store.get_sources()
