# LLMChat

A simple LLM-powered chatbot built as the first project of an AI Engineering course. Connects to free models via [OpenRouter](https://openrouter.ai), with conversation memory, automatic history summarization, token-by-token streaming, and a token budget limit.

## Features

- **LLM connection** via OpenRouter (OpenAI-compatible API)
- **Conversation memory** — full chat history is injected into every request
- **Automatic summarization** — once history grows past a threshold, the older half is summarized by the model itself and replaced with a single system note, keeping token usage bounded
- **Token-by-token streaming** of the response as it's generated
- **Semantic response caching** — similar questions within the same session reuse a cached answer instead of making a redundant API call, which reduces latency and cost while preserving conversation context
- **Token counting & budget limit** — an offline estimate (no external tokenizer calls) tracks usage per session and blocks new requests once a configured budget is exceeded
- **Local RAG** — upload `.txt` and `.md` documents, retrieve relevant chunks with local multilingual embeddings, and answer with numbered source citations

## Setup

Requires Python 3.10+ and [pipenv](https://pipenv.pypa.io/).

```bash
pipenv install
```

Copy `.env.example` to `.env`, add your OpenRouter API key, and optionally select an embedding model:

```bash
cp .env.example .env
```

Get a free key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
`EMBEDDING_MODEL` defaults to `intfloat/multilingual-e5-small`. Transformers downloads that model on first use; document embeddings and retrieval then run locally.

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
- `POST /rag/upload` — ingests a UTF-8 `.txt` or `.md` file (maximum 1 MB) as local searchable chunks
- `GET /rag/sources` — lists indexed filenames and chunk counts
- `DELETE /rag/sources/{name}` — deletes all indexed chunks for a filename

RAG answers include cited sources in the chat stream when matching context is found. Uploaded documents and vector data are persisted under `rag/data/`; that local data directory is git-ignored.

## Tests

Tests use deterministic fake embeddings and do not download models or call the network:

```bash
pipenv run pytest
```

To run just the RAG coverage:

```bash
pipenv run pytest tests/test_rag.py
```

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
rag/                   Local embedding, chunking, vector storage, and retrieval
  embeddings.py        Transformers-based local multilingual embeddings
  chunking.py          Persian/English-aware character chunking
  vector_store.py      Swappable VectorStore interface and NumPy implementation
  retriever.py         Document ingestion, persistence, and top-k retrieval
  data/                Local vectors and metadata (created at runtime)
agents/                Planned agents, MCP client, and tools
observability/         Planned logging, metrics, and evaluation
deployment/            Planned Docker and CI configuration
tests/                 Test directory
```

## Notes

- The default model is `openrouter/free`, a router that automatically picks among available free models — useful because any single free model can get rate-limited or geo-restricted upstream.
- Token counts are an **offline estimate** (char/word-based blend), not an exact tokenizer count, since the underlying free models don't expose one through this API. When the provider returns real usage data in the stream, that's used instead.

## Roadmap

The current milestone includes the OpenRouter chatbot, token tracking, streaming usage capture, session-based SSE chat, and local RAG. The remaining architecture is scaffolded for incremental implementation:

1. Complete the Phase 1 memory and token-budget modules.
2. Extract session management from the web API.
3. Add structured outputs and tool calling.
4. Add agents, observability, and deployment support.

### Manual RAG check

1. Start the server with `pipenv run uvicorn web.api:app --reload` and open `http://127.0.0.1:8000`.
2. Upload a small UTF-8 `.txt` or `.md` document from the header; confirm it appears under Documents with its chunk count.
3. Ask a question answered in that document and confirm the response contains `[1]`-style citations and a source label beneath the message.
4. Delete the document from the Documents list and confirm it no longer appears in `/rag/sources`.
