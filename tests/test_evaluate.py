import pytest

from absa.base import AspectSentiment as AS
from absa.datasets import GoldAspect, GoldSentence, load_semeval
from absa.evaluate import evaluate, match_aspects, normalize


def gold(text, *aspects):
    return GoldSentence(id="x", text=text, aspects=[GoldAspect(t, p, (0, 0)) for t, p in aspects])


def test_normalize():
    assert normalize("The Service!") == "service"
    assert normalize("  their decaf  latte ") == "decaf latte"


def test_match_exact_before_partial():
    pairs = match_aspects(["service", "chocolate cake"], ["cake", "the service"])
    assert sorted(pairs) == [(0, 1, "exact"), (1, 0, "partial")]


def test_match_is_one_to_one():
    assert len(match_aspects(["pizza"], ["pizza", "pizza crust"])) == 1


def test_perfect_prediction():
    g = [gold("t", ("food", "positive"), ("service", "negative"))]
    p = [[AS("food", "positive", 0.9), AS("service", "negative", 0.8)]]
    r = evaluate(g, p)
    assert r.aspect_exact["f1"] == r.pair_f1["f1"] == r.sentiment_accuracy == 1.0


def test_wrong_polarity_hits_pair_f1_only():
    r = evaluate([gold("t", ("food", "positive"))], [[AS("food", "negative", 0.9)]])
    assert r.aspect_exact["f1"] == 1.0
    assert r.pair_f1["f1"] == 0.0
    assert r.confusion["positive"]["negative"] == 1


def test_partial_match_counts_for_partial_and_sentiment_not_pair():
    r = evaluate([gold("t", ("chicken tikka masala", "positive"))], [[AS("masala", "positive", 0.9)]])
    assert r.aspect_exact["f1"] == 0.0
    assert r.aspect_partial["f1"] == 1.0
    assert r.sentiment_accuracy == 1.0
    assert r.pair_f1["f1"] == 0.0


def test_false_positive_and_miss():
    r = evaluate(
        [gold("t", ("food", "positive"), ("decor", "neutral"))],
        [[AS("food", "positive", 1), AS("wine", "positive", 1)]],
    )
    assert r.aspect_exact == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_conflict_terms_are_ignored():
    g = [gold("t", ("food", "conflict"), ("service", "positive"))]
    r = evaluate(g, [[AS("food", "positive", 1), AS("service", "positive", 1)]])
    assert r.n_gold == 1 and r.n_pred == 1
    assert r.pair_f1["f1"] == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate([gold("t")], [])


def test_load_semeval(tmp_path):
    xml = tmp_path / "s.xml"
    xml.write_text(
        '<sentences><sentence id="1"><text>The bread is top notch.</text><aspectTerms>'
        '<aspectTerm term="bread" polarity="positive" from="4" to="9"/></aspectTerms></sentence>'
        '<sentence id="2"><text>Nice.</text></sentence></sentences>',
        encoding="utf-8",
    )
    s = load_semeval(xml)
    assert [len(x.aspects) for x in s] == [1, 0]
    assert s[0].aspects[0] == GoldAspect("bread", "positive", (4, 9))
