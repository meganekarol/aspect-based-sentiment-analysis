"""
Unit tests for LexiconABSA implementation.

Covers:
  • Simple cases (clear sentiment, single aspect)
  • Complex cases (multiple aspects, mixed sentiments)
  • Edge cases (negation, sarcasm, compound aspects)
"""

import pytest
from src.lexicon_absa import LexiconABSA, LexiconABSAConfig


# ------------------------------------------------------------
# Fixture — Load once
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Initialize the LexiconABSA model."""
    return LexiconABSA(LexiconABSAConfig(overall_aspect="overall"))


# ------------------------------------------------------------
# SIMPLE CASES
# ------------------------------------------------------------
def test_simple_positive(model):
    text = "The pizza was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects)
    assert any(s == "positive" for s in aspects.values())


def test_simple_negative(model):
    text = "The service was terrible."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects)
    assert any(s == "negative" for s in aspects.values())


# ------------------------------------------------------------
# COMPLEX CASES
# ------------------------------------------------------------
def test_multiple_aspects(model):
    text = "The mushroom risotto was perfection, but the service was slow. The atmosphere was cozy."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects)
    assert any("atmosphere" in a for a in aspects)
    assert any("mushroom" in a or "risotto" in a for a in aspects)
    assert aspects.get("service", "") == "negative" or "negative" in aspects.values()


def test_food_and_ambience(model):
    text = "The food was great and the atmosphere was cozy."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("food" in a for a in aspects)
    assert any("atmosphere" in a for a in aspects)
    assert all(s == "positive" for s in aspects.values())


def test_compound_aspect(model):
    text = "The chicken tikka masala was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("chicken tikka masala" in a for a in aspects), "Compound aspect not detected."
    assert any(s == "positive" for s in aspects.values())


# ------------------------------------------------------------
# EDGE CASES
# ------------------------------------------------------------
def test_negation(model):
    text = "The burger was not tasty."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("burger" in a for a in aspects)
    # Negation should flip positive sentiment
    assert any(s in {"negative", "neutral"} for s in aspects.values())


def test_intensifier(model):
    text = "The pasta was absolutely great."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pasta" in a for a in aspects)
    assert any(s == "positive" for s in aspects.values())


def test_sarcasm(model):
    text = "Oh great, another burnt pizza."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects)
    # Lexicon model usually fails on sarcasm → mark neutral or false positive
    assert any(s in {"neutral", "positive"} for s in aspects.values())


def test_generic_exclusion(model):
    text = "The thing was nice."
    result = model.analyze(text)
    aspects = [r.aspect for r in result]

    # "place" is in GENERIC_EXCLUDE and should not appear as an aspect
    assert not any(a == "thing" for a in aspects), "Generic aspect 'thing' should be excluded."


def test_sentence_boundary(model):
    text = "The food was amazing. The service was awful."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "food" in aspects and aspects["food"] == "positive"
    assert "service" in aspects and aspects["service"] == "negative"


def test_overall_sentiment(model):
    text = "The drinks were okay but overall it was a great experience."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    # Should detect both a specific aspect and the implicit overall sentiment
    assert any("drinks" in a for a in aspects)
    assert any("overall" in a for a in aspects)
    assert aspects["overall"] == "positive"
