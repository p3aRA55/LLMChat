import asyncio
import io
import json

import pytest
from fastapi import HTTPException, UploadFile

from web import api


class FakeRetriever:
    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    def ingest_text(self, text: str, source: str) -> int:
        self.documents[source] = text
        return 1


def test_upload_accepts_supported_file_and_sanitizes_name(monkeypatch) -> None:
    retriever = FakeRetriever()
    monkeypatch.setattr(api, "get_retriever", lambda: retriever)
    upload = UploadFile(filename="my document.MD", file=io.BytesIO(b"hello"))

    response = asyncio.run(api.rag_upload(upload))

    assert response.source == "my_document.md"
    assert response.chunks == 1
    assert retriever.documents == {"my_document.md": "hello"}


@pytest.mark.parametrize("filename", ["../private.txt", r"..\\private.txt"])
def test_upload_rejects_path_traversal(monkeypatch, filename) -> None:
    monkeypatch.setattr(api, "get_retriever", lambda: FakeRetriever())
    upload = UploadFile(filename=filename, file=io.BytesIO(b"ignored"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.rag_upload(upload))

    assert error.value.status_code == 400


def test_upload_rejects_oversized_file(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_retriever", lambda: FakeRetriever())
    upload = UploadFile(filename="large.txt",
                        file=io.BytesIO(b"x" * 1_000_001))

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.rag_upload(upload))

    assert error.value.status_code == 413


def test_chat_stream_emits_json_sources_before_done(monkeypatch) -> None:
    class FakeBot:
        def stream_reply(self, _message: str):
            yield "answer"
            yield '\n__SOURCES__[{"source":"facts.txt","text":"fact","score":1.0}]__SOURCES__'

    monkeypatch.setattr(api, "get_or_create_session",
                        lambda _id: ("id", FakeBot()))
    response = api.chat(api.ChatRequest(message="question"))

    async def collect() -> str:
        return "".join([part async for part in response.body_iterator])

    stream = asyncio.run(collect())

    assert "event: sources" in stream
    source_event = stream.split("event: sources\ndata: ", 1)[
        1].split("\n\n", 1)[0]
    assert json.loads(source_event)[0]["source"] == "facts.txt"
    assert stream.index("event: sources") < stream.index("event: done")
