"""
Unit tests for the final hybrid TransformerABSA implementation.

This model combines:
    • Aspect extraction via yangheng/deberta-v3-base-end2end-absa
    • Sentiment scoring via yangheng/deberta-v3-base-absa-v1.1

Covers:
    • Simple cases (single aspect)
    • Complex cases (multiple aspects)
    • Edge cases (negation, sarcasm, implicit sentiment)
"""

import pytest
from src.transformer_absa import TransformerABSA, TransformerABSAConfig


# ------------------------------------------------------------
# Fixture — load once
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Load the hybrid TransformerABSA model once for all tests to improve efficiency."""
    return TransformerABSA(TransformerABSAConfig())


# ------------------------------------------------------------
# SIMPLE CASES
# ------------------------------------------------------------
def test_single_positive(model):
    """Test that a clearly positive statement is correctly classified as positive."""
    text = "The pizza was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects), "Expected 'pizza' aspect not found."
    assert any(s == "positive" for s in aspects.values()), "Expected positive sentiment."


def test_single_negative(model):
    """Test that a clearly negative statement is correctly classified as negative."""
    text = "The service was terrible."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects), "Expected 'service' aspect not found."
    assert any(s == "negative" for s in aspects.values()), "Expected negative sentiment."


# ------------------------------------------------------------
# COMPLEX CASES
# ------------------------------------------------------------
def test_multiple_aspects(model):
    """Test detection and correct sentiment assignment for multiple aspects in one review."""
    text = (
        "The mushroom risotto was perfection, but the service was slow. "
        "The atmosphere was cozy and the chocolate cake divine."
    )
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    expected = ["mushroom risotto", "service", "atmosphere", "chocolate cake"]
    for exp in expected:
        assert any(exp in a for a in aspects), f"Missing expected aspect: {exp}"

    # Verify sentiment directions
    assert any("service" in a and aspects[a] == "negative" for a in aspects), "Expected 'service' negative sentiment."
    assert any("mushroom" in a and aspects[a] in {"positive", "neutral"} for a in aspects)
    assert any("chocolate" in a and aspects[a] in {"positive", "neutral"} for a in aspects)


def test_food_and_ambience(model):
    """Test that multiple positive aspects (food and atmosphere) are correctly identified as positive."""
    text = "The food was great and the atmosphere was cozy."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("food" in a for a in aspects), "Expected 'food' aspect not found."
    assert any("atmosphere" in a for a in aspects), "Expected 'atmosphere' aspect not found."
    assert any(s == "positive" for s in aspects.values()), "Expected positive sentiments overall."


# ------------------------------------------------------------
# EDGE CASES
# ------------------------------------------------------------
def test_negation(model):
    """Test that negations like 'not tasty' correctly invert or weaken the sentiment polarity."""
    text = "The burger was not tasty."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("burger" in a for a in aspects), "Expected 'burger' aspect not found."
    assert any(s in {"negative", "neutral"} for s in aspects.values()), "Negation not handled correctly."


def test_sarcasm(model):
    """Test that sarcastic expressions are reasonably interpreted (neutral or mild negative sentiment)."""
    text = "Oh perfect, another burnt pizza. Amazing job."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects), "Expected 'pizza' aspect not found."
    pizza_sents = [s for a, s in aspects.items() if "pizza" in a]
    # Transformers generally handle sarcasm better than lexicon models
    assert any(s in {"neutral", "negative"} for s in pizza_sents), "Sarcasm not handled reasonably."


def test_implicit_aspect(model):
    """Test detection of implicit sentiments without explicit aspect terms (e.g., 'Totally worth every penny')."""
    text = "Totally worth every penny!"
    result = model.analyze(text)

    assert len(result) > 0, "Should capture implicit positive sentiment."
    assert any(r.sentiment == "positive" for r in result), "Expected positive sentiment for implicit aspect."


def test_compound_items(model):
    """Test detection and sentiment assignment for multiple compound aspects in one sentence."""
    text = "I loved the chicken tikka masala and cheese pizza!"
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("chicken tikka masala" in a for a in aspects), "Expected 'chicken tikka masala' aspect not found."
    assert any("cheese pizza" in a for a in aspects), "Expected 'cheese pizza' aspect not found."
    assert all(s == "positive" for s in aspects.values()), "Expected all aspects to be positive."


def test_multiple_sentences(model):
    """Test sentiment extraction across multiple sentences with distinct aspect sentiments."""
    text = (
        "The coffee was strong but the croissant was dry. "
        "Loved the outdoor seating though!"
    )
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("coffee" in a for a in aspects), "Expected 'coffee' aspect not found."
    assert any("croissant" in a for a in aspects), "Expected 'croissant' aspect not found."
    assert any("outdoor seating" in a for a in aspects), "Expected 'outdoor seating' aspect not found."
    assert any(aspects[a] == "negative" for a in aspects if "croissant" in a), "Expected 'croissant' negative sentiment."
    assert any(aspects[a] == "positive" for a in aspects if "outdoor" in a), "Expected 'outdoor' positive sentiment."
