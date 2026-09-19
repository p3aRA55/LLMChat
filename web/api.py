import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.chat_bot import ChatBot
from fastapi.staticfiles import StaticFiles


app = FastAPI()

sessions: dict[str, ChatBot] = {}


def get_or_create_session(
    session_id: str | None = None,
) -> tuple[str, ChatBot]:

    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = ChatBot()

    return new_id, sessions[new_id]


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    session_id, bot = get_or_create_session(req.session_id)

    def event_stream():
        yield f"event: session\ndata: {session_id}\n\n"

        for chunk in bot.stream_reply(req.message):
            safe_chunk = chunk.replace("\n", "\\n")
            yield f"data: {safe_chunk}\n\n"

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


app.mount("/", StaticFiles(directory="web/static", html=True), name="static")
