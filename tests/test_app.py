"""Smoke test for the Streamlit demo (lexicon only, no model downloads)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ABSA_DISABLE_TRANSFORMER", "1")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()


def test_app_renders_example_without_errors(app):
    assert not app.exception
    assert app.radio[0].options == ["Lexicon"]
    table = app.dataframe[0].value
    assert {"Aspect", "Sentiment", "Confidence"} <= set(table.columns)
    assert len(table) > 0


def test_app_analyzes_custom_text(app):
    app.text_area[0].set_value("The food was not bad but the waiter was rude.").run()
    assert not app.exception
    table = app.dataframe[0].value
    sentiments = dict(zip(table["Aspect"], table["Sentiment"], strict=True))
    assert "positive" in sentiments["food"]
    assert "negative" in sentiments["waiter"]
