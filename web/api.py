import json
import re
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.chat_bot import ChatBot
from rag.retriever import Retriever

app = FastAPI()

sessions: dict[str, ChatBot] = {}
MAX_UPLOAD_BYTES = 1_000_000

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_or_create_session(
    session_id: str | None = None,
) -> tuple[str, ChatBot]:

    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = ChatBot(retriever=get_retriever())

    return new_id, sessions[new_id]


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str | None = None


class IngestResponse(BaseModel):
    chunks: int
    source: str


class SourcesResponse(BaseModel):
    sources: dict[str, int]


@app.post("/chat")
def chat(req: ChatRequest):
    session_id, bot = get_or_create_session(req.session_id)

    def event_stream():
        yield f"event: session\ndata: {session_id}\n\n"

        sources_emitted = False
        for chunk in bot.stream_reply(req.message):
            if chunk.startswith("\n__SOURCES__") and chunk.endswith("__SOURCES__"):
                sources_data = chunk[len("\n__SOURCES__") : -len("__SOURCES__")]
                sources = json.loads(sources_data)
                yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
                sources_emitted = True
                continue

            safe_chunk = chunk.replace("\n", "\\n")
            yield f"data: {safe_chunk}\n\n"

        if not sources_emitted:
            yield "event: sources\ndata: []\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.post("/reset")
def reset(req: SessionRequest):
    existed = req.session_id in sessions
    sessions.pop(req.session_id, None)

    return {
        "session_id": req.session_id,
        "reset": existed,
    }


@app.get("/report/{session_id}")
def report(session_id: str):
    bot = sessions.get(session_id)

    if bot is None:
        return {"error": "Session not found"}

    return {
        "session_id": session_id,
        "total_tokens": bot.total_tokens,
        "input_tokens": bot.input_tokens,
        "last_intent": bot.last_intent,
        "output_tokens": bot.output_tokens,
        "estimated_cost_usd": round(
            (bot.input_tokens * 0.05 + bot.output_tokens * 0.08) / 1_000_000,
            6,
        ),
    }


@app.post("/rag/upload", response_model=IngestResponse)
async def rag_upload(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename")

    if (
        "/" in filename
        or "\\" in filename
        or any(part in {".", ".."} for part in re.split(r"[/\\]", filename))
        or any(ord(character) < 32 for character in filename)
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in {"txt", "md"}:
        raise HTTPException(status_code=400, detail="Only .txt and .md files allowed")
    stem = filename[: -(len(extension) + 1)]
    safe_stem = re.sub(r"[^\w-]+", "_", stem, flags=re.UNICODE).strip("._-")[:100]
    if not safe_stem:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = f"{safe_stem}.{extension}"

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 1 MB)")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded") from exc

    try:
        chunks = get_retriever().ingest_text(text, safe_name)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(chunks=chunks, source=safe_name)


@app.get("/rag/sources", response_model=SourcesResponse)
def rag_sources():
    retriever = get_retriever()
    return SourcesResponse(sources=retriever.list_sources())


@app.delete("/rag/sources/{name}")
def rag_delete_source(name: str):
    retriever = get_retriever()
    deleted = retriever.delete_source(name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"deleted": deleted, "source": name}


app.mount("/", StaticFiles(directory="web/static", html=True), name="static")