from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[tuple[dict[str, Any], float]]:
        pass

    @abstractmethod
    def save(self, path: str | Path = "rag/data") -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path = "rag/data") -> None:
        pass

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        pass

    @abstractmethod
    def get_sources(self) -> dict[str, int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class NumpyVectorStore(VectorStore):
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._metadatas: list[dict[str, Any]] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if not np.isfinite(vectors).all():
            raise ValueError("vectors must contain only finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("vectors must not be zero")
        return vectors / norms

    def add(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        normalized = self._normalize(vectors)
        if len(normalized) != len(metadatas):
            raise ValueError("vectors and metadatas must have same length")
        if len(normalized) == 0:
            return
        if self._vectors is not None and normalized.shape[1] != self._vectors.shape[1]:
            raise ValueError("vector dimensions must match the existing store")
        self._vectors = (
            normalized
            if self._vectors is None
            else np.vstack((self._vectors, normalized))
        )
        self._metadatas.extend(dict(metadata) for metadata in metadatas)

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[tuple[dict[str, Any], float]]:
        if top_k <= 0 or self._vectors is None or not len(self._vectors):
            return []
        query = self._normalize(np.asarray(query_vector).reshape(1, -1))[0]
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError("query vector dimension does not match the store")
        scores = self._vectors @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self._metadatas[i], float(scores[i])) for i in indices]

    def save(self, path: str | Path = "rag/data") -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        vectors = (
            self._vectors if self._vectors is not None else np.empty((0, 0), np.float32)
        )
        np.save(directory / "vectors.npy", vectors)
        with (directory / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(self._metadatas, metadata_file, ensure_ascii=False)

    def load(self, path: str | Path = "rag/data") -> None:
        directory = Path(path)
        vectors_path = directory / "vectors.npy"
        metadata_path = directory / "metadata.json"
        if not vectors_path.exists() and not metadata_path.exists():
            self._vectors = None
            self._metadatas = []
            return
        if not vectors_path.exists() or not metadata_path.exists():
            raise ValueError("vector store persistence files are incomplete")

        vectors = np.load(vectors_path, allow_pickle=False)
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadatas = json.load(metadata_file)
        if not isinstance(metadatas, list) or any(
            not isinstance(metadata, dict) for metadata in metadatas
        ):
            raise ValueError("stored metadata must be a list of objects")
        if vectors.ndim != 2 or len(vectors) != len(metadatas):
            raise ValueError("stored vectors and metadata are inconsistent")
        if len(vectors):
            vectors = self._normalize(vectors)
        self._vectors = vectors if len(vectors) else None
        self._metadatas = metadatas

    def delete_by_source(self, source: str) -> int:
        keep = [
            index
            for index, metadata in enumerate(self._metadatas)
            if metadata.get("source") != source
        ]
        deleted = len(self._metadatas) - len(keep)
        if deleted:
            self._metadatas = [self._metadatas[index] for index in keep]
            if self._vectors is not None:
                self._vectors = self._vectors[keep]
                if not len(self._vectors):
                    self._vectors = None
        return deleted

    def get_sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metadata in self._metadatas:
            source = metadata.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._metadatas)
