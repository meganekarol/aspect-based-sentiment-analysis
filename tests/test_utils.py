import pytest
import spacy

from absa.utils import choose_label, extract_aspects_spacy, squash_confidence


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.mark.parametrize(
    ("score", "label"),
    [(0.5, "positive"), (-0.5, "negative"), (0.0, "neutral"), (0.05, "neutral"), (-0.05, "neutral")],
)
def test_choose_label_default_thresholds(score, label):
    assert choose_label(score) == label


def test_choose_label_custom_thresholds():
    assert choose_label(0.1, pos_th=0.12, neg_th=-0.12) == "neutral"
    assert choose_label(0.13, pos_th=0.12, neg_th=-0.12) == "positive"


def test_squash_confidence_range_and_symmetry():
    assert squash_confidence(0.0) == 0.0
    for x in (0.1, 0.5, 1.0):
        c = squash_confidence(x)
        assert 0.0 < c <= 1.0
        assert c == squash_confidence(-x)
    assert squash_confidence(0.2) < squash_confidence(0.8)


def test_extract_aspects_builds_compounds_and_skips_generic(nlp):
    doc = nlp("The customer service was slow but the thing is the pizza was great.")
    aspects = {text for text, _ in extract_aspects_spacy(doc).values()}
    assert "customer service" in aspects
    assert "pizza" in aspects
    assert "thing" not in aspects


def test_extract_aspects_spans_point_into_text(nlp):
    text = "The pizza was delicious."
    for aspect, (start, end) in extract_aspects_spacy(nlp(text)).values():
        assert aspect.split()[-1] in text[start:end].lower()
