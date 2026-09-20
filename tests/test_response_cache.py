from core.response_cache import ResponseCache


def test_lookup_returns_cached_response_for_similar_question():
    cache = ResponseCache()
    cache.store("How do I fix a Python TypeError?",
                "Check the variable type before using it.")

    assert cache.lookup(
        "What is the best way to resolve a Python TypeError?") == "Check the variable type before using it."


def test_lookup_returns_none_for_different_question():
    cache = ResponseCache()
    cache.store("How do I fix a Python TypeError?",
                "Check the variable type before using it.")

    assert cache.lookup("What is the capital of France?") is None
