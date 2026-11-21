"""Cross-model contract test: every back-end honours the shared ``ABSAAnalyzer`` API.

Needs the Hugging Face models and a running Ollama server, so it is excluded
from the default CI run (``-m "not slow and not llm"``).
"""

import pytest

from absa import SENTIMENTS, AspectSentiment, get_analyzer

pytestmark = [pytest.mark.slow, pytest.mark.llm]

LONG_REVIEW = """
Food was good tasting, portions felt small - especially side dishes. I guess they are trying to do
gourmet comfort food, but something didn't mesh. It was sorta family style served, but we felt bad
taking more than a tablespoon of any dish because there wouldn't be enough to share.
"""


@pytest.mark.parametrize("name", ["lexicon", "transformer", "llm"])
def test_analyzer_contract(name):
    results = get_analyzer(name).analyze(LONG_REVIEW)
    assert results, f"{name} found no aspects"
    for r in results:
        assert isinstance(r, AspectSentiment)
        assert r.aspect.strip()
        assert r.sentiment in SENTIMENTS
        assert 0.0 <= r.confidence <= 1.0
        if r.text_span is not None:
            start, end = r.text_span
            assert 0 <= start < end <= len(LONG_REVIEW)
