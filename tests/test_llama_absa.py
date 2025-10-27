"""
Unit tests for LLMABSA (Ollama-backed ABSA).

These tests:
  • Mock `ollama.chat` to avoid requiring a local model
  • Validate parsing, normalization, span finding, and retry logic
  • Cover well-formed JSON, wrapped JSON, and malformed outputs
"""

import json
import pytest

# Adjust the import below if your file/module name differs
from src.llm_absa import LLMABSA


# -------------------------------------------------------------------
# Utilities: a tiny mock for ollama.chat that returns sequential outputs
# -------------------------------------------------------------------
class ChatMock:
    """
    Simple callable that mimics `ollama.chat`.
    It returns sequential canned responses on each call.
    """
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, model, messages, format, options):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return {"message": {"content": self.responses[idx]}}


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def model():
    """Create an LLMABSA instance with fast retries (patched sleep in tests)."""
    return LLMABSA(model_name="llama3", max_retries=3, retry_delay=0.0)


@pytest.fixture
def no_sleep(monkeypatch):
    """Disable time.sleep to make retry tests run instantly."""
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
def test_simple_positive(monkeypatch, model):
    """It should parse a straightforward positive JSON output and return one positive aspect."""
    mock_json = json.dumps([{"aspect": "pizza", "sentiment": "positive", "confidence": 0.92}])
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([mock_json]))

    result = model.analyze("The pizza was delicious.")
    assert len(result) == 1
    r = result[0]
    assert r.aspect == "pizza"
    assert r.sentiment == "positive"

def test_multiple_aspects(monkeypatch, model):
    """It should handle multiple aspects with mixed sentiments."""
    mock_json = json.dumps([
        {"aspect": "service", "sentiment": "negative", "confidence": 0.8},
        {"aspect": "atmosphere", "sentiment": "positive", "confidence": 0.85},
    ])
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([mock_json]))

    text = "Service was slow but the atmosphere was lovely."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}
    assert "service" in aspects and aspects["service"] == "negative"
    assert "atmosphere" in aspects and aspects["atmosphere"] == "positive"


def test_malformed_json_recovery(monkeypatch, model):
    """It should recover from extra prose around the JSON by extracting the JSON array."""
    wrapped = "Here you go:\n" + json.dumps(
        [{"aspect": "burger", "sentiment": "negative", "confidence": 0.9}]
    ) + "\nThanks!"
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([wrapped]))

    result = model.analyze("The burger was burnt.")
    assert len(result) == 1
    assert result[0].aspect == "burger"
    assert result[0].sentiment == "negative"


def test_wrapped_object_with_aspects_key(monkeypatch, model):
    """It should support JSON objects like {'aspects': [...]} (not just arrays)."""
    obj = {"aspects": [
        {"aspect": "decaf latte", "sentiment": "positive", "confidence": 0.8}
    ]}
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([json.dumps(obj)]))

    result = model.analyze("The decaf latte was tasty.")
    assert len(result) == 1
    assert result[0].aspect == "decaf latte"
    assert result[0].sentiment == "positive"


def test_confidence_clamping_and_invalid_sentiment(monkeypatch, model):
    """It should clamp confidence to [0,1] and map unknown sentiments to 'neutral'."""
    mock_json = json.dumps([
        {"aspect": "waiter", "sentiment": "amazing", "confidence": 1.5},  # invalid sentiment & >1.0 confidence
        {"aspect": "parking", "sentiment": "bad", "confidence": -0.3},    # non-standard sentiment & <0.0 confidence
    ])
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([mock_json]))

    res = model.analyze("The waiter was amazing. The parking was bad.")
    sents = {r.aspect: (r.sentiment, r.confidence) for r in res}

    assert sents["waiter"][0] == "neutral" and 0.0 <= sents["waiter"][1] <= 1.0
    assert sents["parking"][0] == "neutral" and 0.0 <= sents["parking"][1] <= 1.0


def test_retry_logic_success_on_third_attempt(monkeypatch, model, no_sleep):
    """
    If the first two outputs are invalid/malformed and the third is valid,
    analyze() should return success after retries.
    """
    bad1 = "NOT JSON AT ALL"
    bad2 = "[]"
    good = json.dumps([{"aspect": "croissant", "sentiment": "negative", "confidence": 0.88}])
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([bad1, bad2, good]))

    res = model.analyze("The croissant was dry.")
    assert len(res) == 1
    assert res[0].aspect == "croissant"
    assert res[0].sentiment == "negative"


def test_all_retries_fail_returns_empty(monkeypatch, model, no_sleep):
    """If all attempts fail to produce parseable JSON, analyze() should return an empty list."""
    monkeypatch.setattr("src.llm_absa.chat", ChatMock(["oops", "still bad", "nope"]))
    res = model.analyze("Text that triggers failures.")
    assert res == []


def test_span_finder_when_aspect_missing(monkeypatch, model):
    """_find_span should return None if the aspect is not found verbatim in text."""
    # Force an aspect that isn't verbatim in the text
    mock_json = json.dumps([{"aspect": "nonexistent aspect", "sentiment": "neutral", "confidence": 0.7}])
    monkeypatch.setattr("src.llm_absa.chat", ChatMock([mock_json]))

    res = model.analyze("Short review.")
    assert len(res) == 1
