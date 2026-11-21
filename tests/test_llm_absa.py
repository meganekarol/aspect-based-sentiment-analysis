import json
from unittest.mock import patch

from absa.base import AspectSentiment
from absa.llm_absa import LLMABSA


# ---------------------------------------------------------------------
# Helper: mock responses from Ollama
# ---------------------------------------------------------------------
def mock_chat_single_aspect(*args, **kwargs):
    return {
        "message": {"content": json.dumps([{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95}])}
    }


def mock_chat_multiple_aspects(*args, **kwargs):
    return {
        "message": {
            "content": json.dumps(
                [
                    {"aspect": "service", "sentiment": "negative", "confidence": 0.85},
                    {"aspect": "food", "sentiment": "positive", "confidence": 0.9},
                    {"aspect": "atmosphere", "sentiment": "neutral", "confidence": 0.7},
                ]
            )
        }
    }


def mock_chat_edge_case(*args, **kwargs):
    # Sarcastic/implicit example
    return {
        "message": {
            "content": json.dumps(
                [
                    {"aspect": "wifi", "sentiment": "negative", "confidence": 0.6},
                    {"aspect": "staff", "sentiment": "positive", "confidence": 0.8},
                ]
            )
        }
    }


def mock_chat_error(*args, **kwargs):
    raise RuntimeError("Local LLM not responding")


# ---------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------


@patch("absa.llm_absa.chat", side_effect=mock_chat_single_aspect)
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


@patch("absa.llm_absa.chat", side_effect=mock_chat_multiple_aspects)
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


@patch("absa.llm_absa.chat", side_effect=mock_chat_edge_case)
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


@patch("absa.llm_absa.chat", side_effect=mock_chat_error)
def test_retry_logic(mock_chat):
    """Test retry mechanism handles repeated failures gracefully."""
    analyzer = LLMABSA(model_name="llama3", retry_delay=0)
    text = "Test retry mechanism when LLM fails."
    results = analyzer.analyze(text)
    assert results == []  # after all retries fail
    assert mock_chat.call_count == 3


@patch("absa.llm_absa.chat")
def test_retries_on_unparseable_output_then_succeeds(mock_chat):
    """Non-JSON output triggers a retry instead of silently returning nothing."""
    mock_chat.side_effect = [
        {"message": {"content": "Sorry, I cannot help with that."}},
        mock_chat_single_aspect(),
    ]
    results = LLMABSA(retry_delay=0).analyze("The pizza was delicious!")
    assert [r.aspect for r in results] == ["pizza"]
    assert mock_chat.call_count == 2


def test_parse_tolerates_wrapping_text_and_dict():
    assert LLMABSA._safe_json_parse('Here you go: [{"aspect": "x"}] done') == [{"aspect": "x"}]
    assert LLMABSA._safe_json_parse('{"aspects": [{"aspect": "x"}]}') == [{"aspect": "x"}]


def test_normalize_validates_labels_and_confidence():
    text = "The Pizza was ok"
    data = [
        {"aspect": "pizza", "sentiment": "Mixed", "confidence": 7},
        {"aspect": "", "sentiment": "positive"},
        "garbage",
        {"aspect": "crust", "sentiment": "negative", "confidence": "high"},
    ]
    results = LLMABSA._normalize_output(data, text)
    assert [(r.aspect, r.sentiment, r.confidence) for r in results] == [
        ("pizza", "neutral", 1.0),
        ("crust", "negative", 0.7),
    ]
    assert results[0].text_span == (4, 9)  # case-insensitive match
    assert results[1].text_span is None
