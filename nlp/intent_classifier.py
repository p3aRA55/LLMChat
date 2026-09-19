from functools import lru_cache

from transformers import pipeline


class IntentClassifier:
    """Zero-shot intent classification for routing chatbot behavior.

    Uses an NLI-based zero-shot pipeline (multilingual model, works with
    Persian input), downloaded once and run locally — no external API call.
    """

    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    LABELS = {
        "technical_question": "سوال فنی یا مفهومی",
        "code_request": "درخواست نوشتن یا اصلاح کد",
        "casual_chat": "گفتگوی عادی و غیرفنی",
        "complaint_or_error": "گزارش خطا یا شکایت",
    }

    DEFAULT_INTENT = "casual_chat"
    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, device: int = -1):
        # device=-1 -> CPU. اگر GPU/CUDA داری بذار 0.
        self._pipe = pipeline(
            "zero-shot-classification",
            model=self.MODEL_NAME,
            device=device,
        )
        self._label_fa_to_key = {v: k for k, v in self.LABELS.items()}

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {"intent": self.DEFAULT_INTENT, "confidence": 0.0, "raw": None}

        candidate_labels = list(self.LABELS.values())
        result = self._pipe(text, candidate_labels=candidate_labels, multi_label=False)

        top_label_fa = result["labels"][0]
        top_score = result["scores"][0]
        intent = self._label_fa_to_key[top_label_fa]

        # زیر آستانه‌ی اطمینان -> به‌جای ریسک missclassification، fallback بزن
        if top_score < self.CONFIDENCE_THRESHOLD:
            intent = self.DEFAULT_INTENT

        return {
            "intent": intent,
            "confidence": top_score,
            "raw": result,
        }


@lru_cache(maxsize=1)
def get_classifier() -> IntentClassifier:
    """Singleton lazy-loader — مدل فقط یک‌بار لود می‌شه، نه هر بار که کلاس‌بندی لازمه."""
    return IntentClassifier()