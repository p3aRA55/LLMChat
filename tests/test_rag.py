from types import SimpleNamespace

import numpy as np

from core.chat_bot import ChatBot
from rag.chunking import chunk_text
from rag.retriever import Retriever
from rag.vector_store import NumpyVectorStore


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]


def test_chunk_text_splits_persian_and_english_boundaries() -> None:
    persian = chunk_text("سلام؟ بله. بعد.", chunk_size=12, overlap=2)
    english = chunk_text("Hello? Yes. Next.", chunk_size=12, overlap=2)

    assert persian == ["سلام؟ بله.", "ه. بعد."]
    assert english == ["Hello? Yes.", "s. Next."]


def test_chunk_text_preserves_overlap_and_respects_size() -> None:
    text = "first sentence. " + ("long words " * 12) + " final sentence."
    chunks = chunk_text(text, chunk_size=32, overlap=8)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 32 for chunk in chunks)
    assert any(
        chunks[index - 1][-8:] == chunks[index][:8]
        for index in range(1, len(chunks))
    )


def test_vector_store_add_search_persistence_and_delete(tmp_path) -> None:
    store = NumpyVectorStore()
    metadata = [
        {"text": "alpha text", "source": "a.txt", "chunk_index": 0},
        {"text": "beta text", "source": "b.txt", "chunk_index": 0},
    ]
    store.add(np.array([[4, 0], [0, 2]], dtype=np.float32), metadata)

    assert np.allclose(np.linalg.norm(store._vectors, axis=1), [1.0, 1.0])
    assert store.search(np.array([1, 0]), 1) == [(metadata[0], 1.0)]
    store.save(tmp_path)

    restored = NumpyVectorStore()
    restored.load(tmp_path)
    assert restored.get_sources() == {"a.txt": 1, "b.txt": 1}
    assert restored.search(np.array([0, 1]), 2)[0][0]["source"] == "b.txt"
    assert restored.delete_by_source("a.txt") == 1
    assert len(restored) == 1


def test_retriever_filters_by_min_score_and_persists(tmp_path) -> None:
    retriever = Retriever(FakeEmbedder(), data_dir=tmp_path)
    retriever.ingest_text("alpha fact.", "alpha.txt")
    retriever.ingest_text("beta fact.", "beta.txt")

    results = retriever.retrieve("alpha", min_score=0.9)
    assert [result["source"] for result in results] == ["alpha.txt"]
    assert retriever.retrieve("alpha", min_score=1.1) == []
    assert retriever.list_sources() == {"alpha.txt": 1, "beta.txt": 1}

    reloaded = Retriever(FakeEmbedder(), data_dir=tmp_path)
    assert reloaded.list_sources() == {"alpha.txt": 1, "beta.txt": 1}


def test_chatbot_injects_request_context_without_persisting_it(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def retrieve(self, _query: str) -> list[dict[str, object]]:
            return [{"text": "context fact", "source": "facts.txt", "score": 1.0}]

        def build_context(self, _results: list[dict[str, object]]) -> str:
            return "[1] context fact (source: facts.txt)"

    class FakeCompletions:
        def create(self, **_kwargs):
            captured["messages"] = _kwargs["messages"]
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="answer"))
                        ],
                        usage=None,
                    )
                ]
            )

    monkeypatch.setattr(
        "core.chat_bot.OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    bot = ChatBot(retriever=FakeRetriever())
    bot.classify_intent = lambda _message: None

    output = list(bot.stream_reply("question"))
    request_messages = captured["messages"]

    assert output[0] == "answer"
    assert "__SOURCES__" in output[1]
    assert any("context fact" in message["content"] for message in request_messages)
    assert all("context fact" not in message["content"] for message in bot.messages)
    assert [message["role"] for message in bot.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert bot.input_tokens == sum(
        bot.count_tokens(message["content"]) for message in request_messages
    )
