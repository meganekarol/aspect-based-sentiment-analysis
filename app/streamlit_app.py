"""Interactive demo: ``streamlit run app/streamlit_app.py``.

Environment switches:
- ``ABSA_DISABLE_TRANSFORMER=1`` hides the Transformer back-end (~1.5 GB RAM).
- ``OLLAMA_HOST=http://host:11434`` enables the LLM back-end.
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
try:
    import absa  # noqa: F401
except ImportError:  # hosted without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from absa import AspectSentiment, get_analyzer  # noqa: E402

EXAMPLES = ROOT / "data" / "examples.jsonl"
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"

MODELS = {
    "lexicon": ("Lexicon", "spaCy dependency rules + VADER lexicon. Fast and explainable."),
    "transformer": ("Transformer", "Two DeBERTa-v3 models: aspect tagging + aspect-conditioned sentiment."),
    "llm": ("LLM", "Few-shot prompted local LLM via Ollama, strict JSON output."),
}
COLORS = {  # translucent so they work on light and dark themes
    "positive": "rgba(31, 158, 116, 0.28)",
    "negative": "rgba(214, 64, 69, 0.28)",
    "neutral": "rgba(128, 128, 128, 0.25)",
}
ICONS = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


def available_models() -> list[str]:
    models = ["lexicon"]
    if os.environ.get("ABSA_DISABLE_TRANSFORMER") != "1":
        models.append("transformer")
    if os.environ.get("OLLAMA_HOST"):
        models.append("llm")
    return models


@st.cache_resource(show_spinner="Loading model (first run downloads weights)…")
def load(name: str):
    kwargs = {"model_name": os.environ.get("ABSA_LLM_MODEL", "llama3")} if name == "llm" else {}
    return get_analyzer(name, **kwargs)


@st.cache_data
def load_examples() -> list[dict]:
    if not EXAMPLES.exists():
        return []
    return [json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(name: str, text: str) -> tuple[list[AspectSentiment], float]:
    analyzer = load(name)  # load outside the timer so latency reflects inference only
    t0 = time.perf_counter()
    results = analyzer.analyze(text)
    return results, (time.perf_counter() - t0) * 1000


def highlight(text: str, results: list[AspectSentiment]) -> str:
    """Return the text as HTML with each aspect span colour-coded by sentiment."""
    spans = sorted((r for r in results if r.text_span), key=lambda r: r.text_span[0])
    out, pos = [], 0
    for r in spans:
        start, end = r.text_span
        if start < pos:  # skip overlapping spans
            continue
        out.append(html.escape(text[pos:start]))
        out.append(
            f'<mark title="{r.sentiment} · {r.confidence:.2f}" style="background:{COLORS[r.sentiment]};'
            f'color:inherit;padding:0.1em 0.25em;border-radius:0.3em">{html.escape(text[start:end])}'
            f'<sub style="font-size:0.65em;opacity:0.75;margin-left:0.25em">{r.sentiment}</sub></mark>'
        )
        pos = end
    out.append(html.escape(text[pos:]))
    return f'<div style="line-height:2.1;font-size:1.05rem">{"".join(out)}</div>'


def results_frame(results: list[AspectSentiment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Aspect": r.aspect,
                "Sentiment": f"{ICONS[r.sentiment]} {r.sentiment}",
                "Confidence": r.confidence,
            }
            for r in results
        ]
    )


def show_results(text: str, results: list[AspectSentiment], ms: float) -> None:
    if not results:
        st.info("No aspects detected.")
        return
    st.markdown(highlight(text, results), unsafe_allow_html=True)
    st.dataframe(
        results_frame(results),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")
        },
    )
    st.caption(f"{len(results)} aspects · {ms:.0f} ms")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Aspect-Based Sentiment Analysis", page_icon="💬", layout="wide")
st.title("Aspect-Based Sentiment Analysis")
st.markdown(
    "Finds **what** a review talks about and **how** the writer feels about each thing, "
    "using three interchangeable approaches behind one API."
)

models = available_models()
examples = load_examples()

with st.sidebar:
    st.header("Input")
    titles = ["(write your own)"] + [e["title"] for e in examples]
    choice = st.selectbox("Example review", titles, index=1 if examples else 0)
    default = examples[titles.index(choice) - 1]["text"] if choice != titles[0] else ""
    text = st.text_area("Review text", value=default, height=180, max_chars=2000)
    st.divider()
    st.caption("Available models: " + ", ".join(MODELS[m][0] for m in models))
    if "llm" not in models:
        st.caption("The LLM back-end needs a local Ollama server; run the app locally to try it.")

tab_one, tab_compare, tab_bench = st.tabs(["Analyze", "Compare models", "Benchmark"])

with tab_one:
    name = st.radio(
        "Model",
        models,
        format_func=lambda m: MODELS[m][0],
        horizontal=True,
        captions=[MODELS[m][1] for m in models],
    )
    if text.strip():
        results, ms = run(name, text)
        show_results(text, results, ms)
    else:
        st.write("Enter a review in the sidebar.")

with tab_compare:
    if not text.strip():
        st.write("Enter a review in the sidebar.")
    elif st.button("Run all available models", type="primary") or st.session_state.get("compared") == text:
        st.session_state["compared"] = text
        cols = st.columns(len(models))
        for col, m in zip(cols, models, strict=True):
            with col:
                st.subheader(MODELS[m][0])
                results, ms = run(m, text)
                show_results(text, results, ms)

with tab_bench:
    md = RESULTS / "benchmark.md"
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))
        for img in ("benchmark_scores.png", "benchmark_tradeoff.png", "benchmark_confusion.png"):
            if (IMG / img).exists():
                st.image(str(IMG / img))
    else:
        st.info("Run `python scripts/benchmark.py` to generate benchmark results.")
