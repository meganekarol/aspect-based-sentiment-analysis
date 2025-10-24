"""
Unit tests for the final hybrid TransformerABSA implementation.
This combines:
    - Aspect extraction via yangheng/deberta-v3-base-end2end-absa
    - Sentiment scoring via yangheng/deberta-v3-base-absa-v1.1

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
    """Load the hybrid TransformerABSA model once for all tests."""
    return TransformerABSA(TransformerABSAConfig())


# ------------------------------------------------------------
# SIMPLE CASES
# ------------------------------------------------------------
def test_single_positive(model):
    text = "The pizza was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects), "Expected 'pizza' aspect not found."
    assert any(s == "positive" for s in aspects.values()), "Expected positive sentiment."


def test_single_negative(model):
    text = "The service was terrible."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects)
    assert any(s == "negative" for s in aspects.values())


# ------------------------------------------------------------
# COMPLEX CASES
# ------------------------------------------------------------
def test_multiple_aspects(model):
    text = (
        "The mushroom risotto was perfection, but the service was slow. "
        "The atmosphere was cozy and the chocolate cake divine."
    )
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    expected = ["mushroom risotto", "service", "atmosphere", "chocolate cake"]
    for exp in expected:
        assert any(exp in a for a in aspects), f"Missing expected aspect: {exp}"

    # Check sentiment directions
    assert any("service" in a and aspects[a] == "negative" for a in aspects), "Expected 'service' negative sentiment."
    assert any("mushroom" in a and aspects[a] in {"positive", "neutral"} for a in aspects)
    assert any("chocolate" in a and aspects[a] in {"positive", "neutral"} for a in aspects)


def test_food_and_ambience(model):
    text = "The food was great and the atmosphere was cozy."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("food" in a for a in aspects)
    assert any("atmosphere" in a for a in aspects)
    assert any(s == "positive" for s in aspects.values())


# ------------------------------------------------------------
# EDGE CASES
# ------------------------------------------------------------
def test_negation(model):
    text = "The burger was not tasty."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("burger" in a for a in aspects)
    assert any(s in {"negative", "neutral"} for s in aspects.values()), "Negation not handled correctly."


def test_sarcasm(model):
    text = "Oh perfect, another burnt pizza. Amazing job."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects), "Expected 'pizza' aspect not found."
    pizza_sents = [s for a, s in aspects.items() if "pizza" in a]
    # Transformers usually interpret sarcasm as neutral or mild negative
    assert any(s in {"neutral", "negative"} for s in pizza_sents), "Sarcasm not handled reasonably."


def test_implicit_aspect(model):
    text = "Totally worth every penny!"
    result = model.analyze(text)

    assert len(result) > 0, "Should capture implicit positive sentiment."
    assert any(r.sentiment == "positive" for r in result), "Expected positive sentiment for implicit aspect."


def test_compound_items(model):
    text = "I loved the chicken tikka masala and cheese pizza!"
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("chicken tikka masala" in a for a in aspects)
    assert any("cheese pizza" in a for a in aspects)
    assert all(s == "positive" for s in aspects.values())


def test_multiple_sentences(model):
    text = (
        "The coffee was strong but the croissant was dry. "
        "Loved the outdoor seating though!"
    )
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("coffee" in a for a in aspects)
    assert any("croissant" in a for a in aspects)
    assert any("outdoor seating" in a for a in aspects)
    assert any(aspects[a] == "negative" for a in aspects if "croissant" in a)
    assert any(aspects[a] == "positive" for a in aspects if "outdoor" in a)
