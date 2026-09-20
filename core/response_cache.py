import numpy as np


class ResponseCache:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.entries = []

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _embed(self, query: str):
        if not query:
            return np.zeros(384, dtype=np.float32)
        model = self._get_model()
        embedding = model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embedding, dtype=np.float32)

    def _similarity(self, left, right) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def lookup(self, query: str, threshold: float = 0.85):
        if not query or not self.entries:
            return None

        query_embedding = self._embed(query)
        best_match = None
        best_score = -1.0

        for entry in self.entries:
            score = self._similarity(query_embedding, entry['embedding'])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match is not None and best_score >= threshold:
            return best_match['response']
        return None

    def store(self, query: str, response):
        if not query:
            return None
        self.entries.append({
            'query': query,
            'response': response,
            'embedding': self._embed(query),
        })
        return response


if __name__ == '__main__':
    cache = ResponseCache()
    cache.store(
        'How do I fix a Python TypeError?',
        'Check the variable type before using it.'
    )

    similar = cache.lookup(
        'What is the best way to resolve a Python TypeError?')
    different = cache.lookup('What is the capital of France?')

    print('Similar question hit:', similar)
    print('Different question miss:', different)
