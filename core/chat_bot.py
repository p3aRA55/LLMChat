import dotenv
import json
import os
import re
from openai import OpenAI

from core.response_cache import ResponseCache
from nlp.intent_classifier import get_classifier
from rag.retriever import Retriever

dotenv.load_dotenv()


class ChatBot:
    MODEL = 'openrouter/free'
    MAX_HISTORY_MESSAGES = 10
    SUMMARIZE_EVERY = 10
    TOTAL_TOKEN_LIMIT = 10_000

    SYSTEM_PROMPTS = {
        "technical_question": "you are a helpful assistant, Your name is AzChat. "
        "Answer technical questions precisely and cite reasoning briefly.",
        "code_request": "you are a helpful assistant, Your name is AzChat. "
        "Focus on producing correct, runnable code with minimal but essential explanation.",
        "casual_chat": "you are a helpful assistant, Your name is AzChat. "
        "Keep tone casual and answers short.",
        "complaint_or_error": "you are a helpful assistant, Your name is AzChat. "
        "Be empathetic, ask for error details/logs if missing, and prioritize root-cause diagnosis.",
    }
    DEFAULT_SYSTEM_PROMPT = (
        'you are a helpful assistant, Your name is AzChat, Give short and concise answers.'
    )

    def __init__(self, retriever: Retriever | None = None):
        self.llm = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url='https://openrouter.ai/api/v1'
        )
        self.messages = [
            {
                'role': 'system',
                'content': self.DEFAULT_SYSTEM_PROMPT
            }
        ]
        self.response_cache = ResponseCache()
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_intent = None
        self.retriever = retriever

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        char_estimate = len(text) / 4
        word_estimate = len(text.split()) * 1.3
        return max(1, round((char_estimate + word_estimate) / 2))

    def classify_intent(self, message: str):
        try:
            classifier = get_classifier()
            result = classifier.classify(message)
            self.last_intent = result["intent"]
            return result["intent"]
        except Exception as e:
            print(f"\n⚠️ intent classification skipped, error: {e}")
            self.last_intent = None
            return None

    def apply_intent_prompt(self, intent: str):
        prompt = self.SYSTEM_PROMPTS.get(intent, self.DEFAULT_SYSTEM_PROMPT)
        for m in self.messages:
            if m['role'] == 'system' and not m['content'].startswith('Summary of earlier conversation:'):
                m['content'] = prompt
                break

    def trim_messages(self, max_length=None):
        max_length = max_length or self.MAX_HISTORY_MESSAGES
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']
        if len(other_messages) <= max_length:
            return
        trimmed_messages = other_messages[-max_length:]
        self.messages = [*system_messages, *trimmed_messages]

    def summarize_messages(self):
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
        base_system_messages = [
            m for m in system_messages
            if not m['content'].startswith('Summary of earlier conversation:')
        ]
        self.messages = [*base_system_messages, summarized_message, *recent_messages]

    def stream_reply(self, message: str):
        intent = self.classify_intent(message)
        if intent:
            self.apply_intent_prompt(intent)

        self.messages.append({'role': 'user', 'content': message})

        cached_response = (
            self.response_cache.lookup(message) if self.retriever is None else None
        )
        if cached_response is not None:
            self.messages.append({'role': 'assistant', 'content': cached_response})
            self.summarize_messages()
            self.trim_messages()
            yield cached_response
            return

        rag_context = ""
        rag_sources: list[dict] = []
        if self.retriever:
            results = self.retriever.retrieve(message)
            if results:
                rag_context = self.retriever.build_context(results)
                rag_sources = results

        rag_system_msg = None
        if rag_context:
            rag_system_msg = {
                'role': 'system',
                'content': (
                    "Answer using the context below. If the answer is not in the context, say so. "
                    "Cite sources as [n].\n\nContext:\n" + rag_context
                )
            }

        prompt_tokens_est = sum(self.count_tokens(m['content']) for m in self.messages)
        if rag_system_msg:
            prompt_tokens_est += self.count_tokens(rag_system_msg['content'])

        if self.total_tokens + prompt_tokens_est > self.TOTAL_TOKEN_LIMIT:
            self.messages.pop()
            yield "⛔ محدودیت توکن جلسه فعلی به پایان رسیده است."
            return

        messages = list(self.messages)
        if rag_system_msg:
            messages.insert(1, rag_system_msg)

        try:
            response = self.llm.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                stream=True,
                extra_body={"usage": {"include": True}}
            )
        except Exception as e:
            self.messages.pop()
            yield f"❌ خطا در ارتباط با API: {str(e)}"
            return

        response_text = ""
        is_thinking = False
        usage = None

        for chunk in response:
            chunk_usage = getattr(chunk, 'usage', None)
            if chunk_usage:
                usage = chunk_usage
            if not chunk.choices or not chunk.choices[0].delta:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if "<?xml" in delta:
                is_thinking = True
                delta = delta.replace("<?xml", "")
            if "?>" in delta:
                is_thinking = False
                delta = delta.replace("?>", "")
            if not is_thinking and delta:
                response_text += delta
                yield delta

        clean_response = re.sub(r'<\?xml.*?\?>', '', response_text, flags=re.DOTALL).strip()
        if clean_response:
            self.messages.append({'role': 'assistant', 'content': clean_response})
            if usage:
                self.total_tokens += usage.total_tokens
                self.input_tokens += usage.prompt_tokens
                self.output_tokens += usage.completion_tokens
            else:
                out_est = self.count_tokens(clean_response)
                self.input_tokens += prompt_tokens_est
                self.output_tokens += out_est
                self.total_tokens += prompt_tokens_est + out_est
            if self.retriever is None:
                self.response_cache.store(message, clean_response)

        if rag_sources:
            yield f"\n__SOURCES__{json.dumps(rag_sources, ensure_ascii=False)}__SOURCES__"

        self.summarize_messages()
        self.trim_messages()

    def bot(self, message: str) -> str:
        full_text = ""
        for chunk in self.stream_reply(message):
            print(chunk, end='', flush=True)
            full_text += chunk
        print()
        return full_text

    def chat(self):
        while True:
            user_input = input(
                'Ask something ... (type "q" to terminate): ').strip()
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
        print(f"Last intent: {self.last_intent}")
        print(
            f"Estimated cost: ${(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000}")


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == "__main__":
    main()