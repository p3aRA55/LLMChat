import os
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        model_name = os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_batch(self, texts: List[str], prefix: str) -> np.ndarray:
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        encoded_input = self._tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode_batch(texts, "passage: ")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_batch([text], "query: ")[0]
