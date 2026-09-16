# LLMChat

A simple LLM-powered chatbot built as the first project of an AI Engineering course. Connects to free models via [OpenRouter](https://openrouter.ai), with conversation memory, automatic history summarization, token-by-token streaming, and a token budget limit.

## Features

- **LLM connection** via OpenRouter (OpenAI-compatible API)
- **Conversation memory** — full chat history is injected into every request
- **Automatic summarization** — once history grows past a threshold, the older half is summarized by the model itself and replaced with a single system note, keeping token usage bounded
- **Token-by-token streaming** of the response as it's generated
- **Token counting & budget limit** — an offline estimate (no external tokenizer calls) tracks usage per session and blocks new requests once a configured budget is exceeded

## Setup

Requires Python 3.10+ and [pipenv](https://pipenv.pypa.io/).

```bash
pipenv install
```

Copy `.env.example` to `.env` and add your OpenRouter API key:

```bash
cp .env.example .env
```

Get a free key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).

## Usage

```bash
pipenv run python core/chat_bot.py
```

Type your message and press Enter. Type `q` to quit; a final token/cost report is printed on exit.

To see which free models are currently available on your OpenRouter account:

```bash
pipenv run python list_models.py
```

To run the web interface:

```bash
pipenv run uvicorn web.api:app --reload
```

Then open `http://127.0.0.1:8000` in a browser. The web API exposes:

- `POST /chat` — streams responses over Server-Sent Events and emits a session ID
- `POST /reset` — resets a session
- `GET /report/{session_id}` — returns session token and cost information

## Project Structure

```text
core/                 Core chatbot logic
  chat_bot.py         OpenRouter streaming chatbot
  memory.py           Planned memory abstraction
  token_budget.py     Planned token-budget abstraction
web/                  FastAPI backend and frontend
  api.py              Chat, reset, and report endpoints
  session_manager.py  Planned session abstraction
  static/index.html   Browser chat interface
structured/            Planned structured output and tool calling
rag/                   Planned embeddings, vector store, and retrieval
agents/                Planned agents, MCP client, and tools
observability/         Planned logging, metrics, and evaluation
deployment/            Planned Docker and CI configuration
tests/                 Test directory
```

## Notes

- The default model is `openrouter/free`, a router that automatically picks among available free models — useful because any single free model can get rate-limited or geo-restricted upstream.
- Token counts are an **offline estimate** (char/word-based blend), not an exact tokenizer count, since the underlying free models don't expose one through this API. When the provider returns real usage data in the stream, that's used instead.

## Roadmap

The current milestone includes the OpenRouter chatbot, token tracking, streaming usage capture, and a minimal FastAPI web interface with session-based SSE chat. The remaining architecture is scaffolded for incremental implementation:

1. Complete the Phase 1 memory and token-budget modules.
2. Extract session management from the web API.
3. Add structured outputs and tool calling.
4. Add RAG, agents, observability, and deployment support.
