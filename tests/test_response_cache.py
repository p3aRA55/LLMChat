from core.response_cache import ResponseCache


def _fake_embed(query):
    import numpy as np

    return np.array([1.0, 0.0] if "TypeError" in query else [0.0, 1.0])


def test_lookup_returns_cached_response_for_similar_question(monkeypatch):
    cache = ResponseCache()
    monkeypatch.setattr(cache, "_embed", _fake_embed)
    cache.store("How do I fix a Python TypeError?",
                "Check the variable type before using it.")

    assert cache.lookup(
        "What is the best way to resolve a Python TypeError?") == "Check the variable type before using it."


def test_lookup_returns_none_for_different_question(monkeypatch):
    cache = ResponseCache()
    monkeypatch.setattr(cache, "_embed", _fake_embed)
    cache.store("How do I fix a Python TypeError?",
                "Check the variable type before using it.")

    assert cache.lookup("What is the capital of France?") is None
