# rag/embeddings.py
from sentence_transformers import SentenceTransformer, util
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def most_similar(self, query: str, candidates: list[str], top_k: int = 1):
        query_vec = self.embed(query)
        candidate_vecs = self.embed(candidates)
        scores = util.cos_sim(query_vec, candidate_vecs)[0]
        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]