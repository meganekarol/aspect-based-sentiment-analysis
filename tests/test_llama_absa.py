import json
import pytest
from unittest.mock import patch
from src.llm_absa import LLMABSA
from src.base import AspectSentiment


# ---------------------------------------------------------------------
# Helper: mock responses from Ollama
# ---------------------------------------------------------------------
def mock_chat_single_aspect(*args, **kwargs):
    return {
        "message": {
            "content": json.dumps([
                {"aspect": "pizza", "sentiment": "positive", "confidence": 0.95}
            ])
        }
    }


def mock_chat_multiple_aspects(*args, **kwargs):
    return {
        "message": {
            "content": json.dumps([
                {"aspect": "service", "sentiment": "negative", "confidence": 0.85},
                {"aspect": "food", "sentiment": "positive", "confidence": 0.9},
                {"aspect": "atmosphere", "sentiment": "neutral", "confidence": 0.7}
            ])
        }
    }


def mock_chat_edge_case(*args, **kwargs):
    # Sarcastic/implicit example
    return {
        "message": {
            "content": json.dumps([
                {"aspect": "wifi", "sentiment": "negative", "confidence": 0.6},
                {"aspect": "staff", "sentiment": "positive", "confidence": 0.8}
            ])
        }
    }


def mock_chat_error(*args, **kwargs):
    raise RuntimeError("Local LLM not responding")


# ---------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------

@patch("src.llm_absa.chat", side_effect=mock_chat_single_aspect)
def test_simple_case(mock_chat):
    """Test a simple review with one clear aspect and sentiment."""
    analyzer = LLMABSA(model_name="llama3")
    text = "The pizza was delicious!"
    results = analyzer.analyze(text)

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, AspectSentiment)
    assert r.aspect == "pizza"
    assert r.sentiment == "positive"
    assert 0.0 <= r.confidence <= 1.0


@patch("src.llm_absa.chat", side_effect=mock_chat_multiple_aspects)
def test_complex_case(mock_chat):
    """Test a review with multiple aspects and mixed sentiments."""
    analyzer = LLMABSA(model_name="llama3")
    text = "The food was amazing but the service was slow and the atmosphere okay."
    results = analyzer.analyze(text)

    aspects = [r.aspect for r in results]
    assert set(aspects) == {"food", "service", "atmosphere"}
    assert any(r.sentiment == "negative" for r in results)
    assert any(r.sentiment == "positive" for r in results)
    assert any(r.sentiment == "neutral" for r in results)


@patch("src.llm_absa.chat", side_effect=mock_chat_edge_case)
def test_edge_case_sarcasm(mock_chat):
    """Test sarcastic or implicit sentiment cases."""
    analyzer = LLMABSA(model_name="llama3")
    text = "The wifi was *so fast* (not). The staff tried their best though."
    results = analyzer.analyze(text)

    # Expect both aspects found
    assert len(results) == 2
    sentiments = {r.aspect: r.sentiment for r in results}
    assert sentiments["wifi"] == "negative"
    assert sentiments["staff"] == "positive"


@patch("src.llm_absa.chat", side_effect=mock_chat_error)
def test_retry_logic(mock_chat):
    """Test retry mechanism handles repeated failures gracefully."""
    analyzer = LLMABSA(model_name="llama3")
    text = "Test retry mechanism when LLM fails."
    results = analyzer.analyze(text)
    assert results == []  # after all retries fail
