import dotenv
import os
import re
from openai import OpenAI

dotenv.load_dotenv()


class ChatBot:

    # openrouter/free: a router that auto-picks among available free models,
    # so if one provider is rate-limited/geo-blocked it falls back to another.
    MODEL = 'openrouter/free'
    MAX_HISTORY_MESSAGES = 10
    SUMMARIZE_EVERY = 10
    TOTAL_TOKEN_LIMIT = 10_000

    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url='https://openrouter.ai/api/v1'
        )

        self.messages = [
            {
                'role': 'system',
                'content': 'you are a helpful assistant, Your name is AzChat, Give short and concise answers.'
            }
        ]

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def count_tokens(self, text: str) -> int:
        """Offline ESTIMATE only (no network calls, no real tokenizer for
        Gemma/others available through this API). Blend of char-based and
        word-based estimate to stay conservative for Persian/UTF-8 text."""
        if not text:
            return 0
        char_estimate = len(text) / 4
        word_estimate = len(text.split()) * 1.3
        return max(1, round((char_estimate + word_estimate) / 2))

    def trim_messages(self, max_length=None):
        """Keep only the most recent `max_length` non-system messages."""
        max_length = max_length or self.MAX_HISTORY_MESSAGES
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        if len(other_messages) <= max_length:
            return

        trimmed_messages = other_messages[-max_length:]
        self.messages = [*system_messages, *trimmed_messages]

    def summarize_messages(self):
        """Summarize the oldest half of the conversation once history grows
        past SUMMARIZE_EVERY non-system messages, replacing it with a single
        system note so tokens don't grow unbounded."""
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        if len(other_messages) < self.SUMMARIZE_EVERY:
            return

        split_point = len(other_messages) // 2
        early_messages, recent_messages = other_messages[:split_point], other_messages[split_point:]

        try:
            response = self.llm.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {'role': 'system', 'content': 'Summarize these messages concisely, keeping key facts and information.'},
                    *early_messages
                ]
            )
        except Exception as e:
            print(f"\n⚠️ summarization skipped, API error: {e}")
            return

        summary_content = response.choices[0].message.content
        summarized_message = {
            'role': 'system',
            'content': f'Summary of earlier conversation: {summary_content}'
        }

        # keep only ONE running summary system message (don't stack them)
        base_system_messages = [
            m for m in system_messages
            if not m['content'].startswith('Summary of earlier conversation:')
        ]

        self.messages = [*base_system_messages, summarized_message, *recent_messages]

    def stream_reply(self, message: str):
        """Core streaming logic as a generator, so both the CLI and the web
        API can consume it the same way. Yields plain text chunks as they
        arrive; updates self.messages / token counters as a side effect once
        the stream is exhausted."""
        self.messages.append({'role': 'user', 'content': message})

        prompt_tokens_est = sum(self.count_tokens(m['content']) for m in self.messages)
        if self.total_tokens + prompt_tokens_est > self.TOTAL_TOKEN_LIMIT:
            self.messages.pop()
            yield "⛔ Token budget exceeded for this session."
            return

        try:
            response = self.llm.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                stream=True,
                extra_body={"usage": {"include": True}}
            )
        except Exception as e:
            self.messages.pop()
            yield f"❌ API connection error: {e}"
            return

        response_text = ""
        is_thinking = False
        usage_chunk = None

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content

                if "<think>" in delta:
                    is_thinking = True
                    delta = delta.replace("<think>", "")
                if "</think>" in delta:
                    is_thinking = False
                    delta = delta.replace("</think>", "")

                if not is_thinking and delta:
                    response_text += delta
                    yield delta

            if getattr(chunk, "usage", None):
                usage_chunk = chunk.usage

        clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        self.messages.append({'role': 'assistant', 'content': clean_response})

        if usage_chunk:
            self.total_tokens += usage_chunk.total_tokens
            self.input_tokens += usage_chunk.prompt_tokens
            self.output_tokens += usage_chunk.completion_tokens
        else:
            out_est = self.count_tokens(clean_response)
            self.input_tokens += prompt_tokens_est
            self.output_tokens += out_est
            self.total_tokens += prompt_tokens_est + out_est

        self.summarize_messages()
        self.trim_messages()

    def bot(self, message):
        """CLI-friendly wrapper: prints as it streams, returns full text."""
        full_text = ""
        for chunk in self.stream_reply(message):
            print(chunk, end='', flush=True)
            full_text += chunk
        print()
        return full_text

    def chat(self):
        while True:
            user_input = input('Ask something ... (type "q" to terminate): ').strip()

            if user_input.lower() == 'q':
                print("Good luck")
                break

            if not user_input:
                continue

            if self.total_tokens > self.TOTAL_TOKEN_LIMIT:
                print("unfortunately you exceeded your token limit today.")
                break

            self.bot(user_input)

    def report(self):
        print("\n--- Final Report ---")
        for message in self.messages:
            print(message)
        print(f"Consumed tokens: {self.total_tokens}")
        print(f"Estimated cost: ${(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000}")


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == "__main__":
    main()
