# AzChat

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
pipenv run python chat_bot.py
```

Type your message and press Enter. Type `q` to quit; a final token/cost report is printed on exit.

To see which free models are currently available on your OpenRouter account:

```bash
pipenv run python list_models.py
```

## Notes

- The default model is `openrouter/free`, a router that automatically picks among available free models — useful because any single free model can get rate-limited or geo-restricted upstream.
- Token counts are an **offline estimate** (char/word-based blend), not an exact tokenizer count, since the underlying free models don't expose one through this API. When the provider returns real usage data in the stream, that's used instead.

## Roadmap

This is project 1 of a multi-part AI Engineering course; a minimal FastAPI web layer is planned as a follow-up so the bot can run outside the CLI.
