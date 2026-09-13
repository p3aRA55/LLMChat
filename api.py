from fastapi import FastAPI
from chat_bot import ChatBot
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = FastAPI()
bot = ChatBot()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat1")
def chat_simple(req: ChatRequest):
    return {"reply": bot.bot(req.message)}

@app.post("/chat2")
def chat_streaming(req: ChatRequest):
    def event_stream():
        for chunk in bot.stream_reply(req.message):
            yield f"Data: {chunk}\n\n"
    return StreamingResponse(event_stream() , media_type="text/event-stream")



