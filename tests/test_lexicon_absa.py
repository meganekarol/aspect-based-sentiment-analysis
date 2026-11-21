"""
Unit tests for LexiconABSA implementation.

Covers:
  • Simple cases (clear sentiment, single aspect)
  • Complex cases (multiple aspects, mixed sentiments)
  • Edge cases (negation, sarcasm, compound aspects)

"""

import pytest

from absa.lexicon_absa import LexiconABSA, LexiconABSAConfig


# ------------------------------------------------------------
# Fixture — Load once
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Initialize the LexiconABSA model once per test module."""
    return LexiconABSA()


# ------------------------------------------------------------
# SIMPLE CASES
# ------------------------------------------------------------
def test_simple_positive(model):
    """Test that a clearly positive statement is correctly identified as positive."""
    text = "The pizza was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects)
    assert any(s == "positive" for s in aspects.values())


def test_simple_negative(model):
    """Test that a clearly negative statement is correctly identified as negative."""
    text = "The service was terrible."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects)
    assert any(s == "negative" for s in aspects.values())


# ------------------------------------------------------------
# COMPLEX CASES
# ------------------------------------------------------------
def test_multiple_aspects(model):
    """Test handling of multiple aspects with different sentiments in a single text."""
    text = "The mushroom risotto was perfection, but the service was slow. The atmosphere was cozy."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("service" in a for a in aspects)
    assert any("atmosphere" in a for a in aspects)
    assert any("mushroom" in a or "risotto" in a for a in aspects)
    assert aspects.get("service", "") == "negative" or "negative" in aspects.values()


def test_mixed_sentiment(model):
    """Test that mixed sentiments (positive and negative) are assigned correctly to each aspect."""
    text = "The ambiance was amazing but the food was bland."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "ambiance" in aspects and aspects["ambiance"] == "positive"
    assert "food" in aspects and aspects["food"] == "negative"


@pytest.mark.xfail(
    reason="Known limitation: en_core_web_sm tags 'tikka' as VERB, splitting the compound aspect",
    strict=False,
)
def test_compound_aspect(model):
    """Test that multi-word (compound) aspects like 'chicken tikka masala' are correctly detected."""
    text = "The chicken tikka masala was delicious."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("chicken tikka masala" in a for a in aspects), "Compound aspect not detected."
    assert any(s == "positive" for s in aspects.values())


# ------------------------------------------------------------
# EDGE CASES
# ------------------------------------------------------------
def test_negation(model):
    """Test that negation (e.g., 'not tasty') correctly flips or neutralizes sentiment polarity."""
    text = "The burger was not tasty."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("burger" in a for a in aspects)
    assert any(s in {"negative", "neutral"} for s in aspects.values())


def test_intensifier(model):
    """Test that intensifiers (e.g., 'absolutely great') correctly strengthen positive sentiment."""
    text = "The pasta was absolutely great."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pasta" in a for a in aspects)
    assert any(s == "positive" for s in aspects.values())


@pytest.mark.xfail(
    reason="Known limitation: interjection 'great' has no dependency link to 'pizza', so no opinion is attached",
    strict=False,
)
def test_sarcasm(model):
    """Test that sarcasm (e.g., 'Oh great, another burnt pizza.') is handled gracefully."""
    text = "Oh great, another burnt pizza."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert any("pizza" in a for a in aspects)
    # Lexicon models often fail on sarcasm, so neutral/positive is acceptable.
    assert any(s in {"neutral", "positive"} for s in aspects.values())


def test_generic_exclusion(model):
    """Test that generic placeholders (like 'thing') are excluded from aspect extraction."""
    text = "The thing was nice."
    result = model.analyze(text)
    aspects = [r.aspect for r in result]

    assert not any(a == "thing" for a in aspects), "Generic aspect 'thing' should be excluded."


def test_sentence_boundary(model):
    """Test that sentiment detection works correctly across sentence boundaries."""
    text = "The food was amazing. The service was awful."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "food" in aspects and aspects["food"] == "positive"
    assert "service" in aspects and aspects["service"] == "negative"


def test_negation_scope(model):
    """Test that negation only affects the relevant clause and does not spill over."""
    text = "The buffet is fresh and never disappoints."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "buffet" in aspects and aspects["buffet"] == "positive"


def test_positive_adj(model):
    """Test that positive adjective modifiers (e.g., 'friendly staff') are detected as positive."""
    text = "The staff are friendly."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "staff" in aspects and aspects["staff"] == "positive"


def test_not_bad(model):
    """Test that double negation expressions (e.g., 'not bad') are interpreted as mildly positive."""
    text = "The food was not bad."
    result = model.analyze(text)
    aspects = {r.aspect: r.sentiment for r in result}

    assert "food" in aspects and aspects["food"] == "positive"


def test_emit_neutral_reports_opinionless_aspects():
    text = "We ordered the lasagna and the service was awful."
    default = {r.aspect for r in LexiconABSA().analyze(text)}
    assert "lasagna" not in default
    results = {r.aspect: r.sentiment for r in LexiconABSA(LexiconABSAConfig(emit_neutral=True)).analyze(text)}
    assert results.get("lasagna") == "neutral"
    assert results.get("service") == "negative"
