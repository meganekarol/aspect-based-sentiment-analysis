"""Benchmark the ABSA back-ends on SemEval-2014 Task 4 (restaurants, gold test set).

    python scripts/benchmark.py --models lexicon transformer
    python scripts/benchmark.py --models llm --llm-model llama3   # needs Ollama running
    python scripts/benchmark.py --report-only                     # rebuild tables/charts

Writes, per model, ``results/<model>.json`` (metrics) and
``results/predictions/<model>.jsonl`` (raw outputs for error analysis), then
regenerates ``results/benchmark.md`` and the charts in ``docs/img/`` from all
model results present, so models can be benchmarked in separate runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from absa import get_analyzer
from absa.datasets import download_semeval, load_semeval
from absa.evaluate import evaluate

RESULTS = Path("results")
IMG = Path("docs/img")
LABELS = {
    "lexicon": "Lexicon (spaCy + VADER)",
    "transformer": "Transformer (DeBERTa-v3)",
    "llm": "LLM (Ollama)",
}
ORDER = ["lexicon", "transformer", "llm"]


def run_model(name: str, sentences, llm_model: str) -> dict:
    kwargs = {"model_name": llm_model} if name == "llm" else {}
    analyzer = get_analyzer(name, **kwargs)
    analyzer.analyze("Warm-up sentence: the food was good.")  # exclude lazy init from timings

    preds, latencies = [], []
    for i, s in enumerate(sentences, 1):
        t0 = time.perf_counter()
        preds.append(analyzer.analyze(s.text))
        latencies.append((time.perf_counter() - t0) * 1000)
        if i % 100 == 0:
            print(f"  [{name}] {i}/{len(sentences)}")

    result = evaluate(sentences, preds).to_dict()
    result.update(
        model=name,
        model_detail=llm_model if name == "llm" else None,
        latency_ms_mean=round(statistics.mean(latencies), 2),
        latency_ms_p95=round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2),
        hardware=f"{platform.processor() or platform.machine()} (CPU)",
    )

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "predictions" / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for s, p in zip(sentences, preds, strict=True):
            row = {
                "id": s.id,
                "text": s.text,
                "gold": [[a.term, a.polarity] for a in s.aspects],
                "pred": [[r.aspect, r.sentiment, r.confidence] for r in p],
            }
            f.write(json.dumps(row) + "\n")
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_results() -> list[dict]:
    return [
        json.loads((RESULTS / f"{m}.json").read_text()) for m in ORDER if (RESULTS / f"{m}.json").exists()
    ]


def write_markdown(results: list[dict]) -> None:
    n = results[0]["n_sentences"]
    lines = [
        "# Benchmark: SemEval-2014 Task 4, restaurants (gold test set)",
        "",
        f"{n} sentences, {results[0]['n_gold']} gold aspect terms (14 `conflict` terms excluded).",
        "",
        "| Model | Pair F1 (strict) | Aspect F1 (exact) | Aspect F1 (partial) | Sentiment acc. | Sentiment macro-F1 | Latency (ms/sentence) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        name = LABELS[r["model"]] + (f" `{r['model_detail']}`" if r.get("model_detail") else "")
        lines.append(
            f"| {name} | **{r['pair_f1']['f1']:.3f}** | {r['aspect_exact']['f1']:.3f} | "
            f"{r['aspect_partial']['f1']:.3f} | {r['sentiment_accuracy']:.3f} | "
            f"{r['sentiment_macro_f1']:.3f} | {r['latency_ms_mean']:.0f} |"
        )
    lines += [
        "",
        "- **Pair F1 (strict)**: aspect string matches exactly *and* polarity is correct.",
        "- **Aspect F1**: aspect extraction only; *partial* = shares a content word with the gold term.",
        "- **Sentiment acc. / macro-F1**: polarity on aspects that were found (partial match).",
        f"- Latency: mean per sentence on {results[0]['hardware']}, after warm-up.",
        "",
        "Precision / recall detail:",
        "",
        "| Model | Aspect P (exact) | Aspect R (exact) | Pair P | Pair R | Predicted aspects |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {LABELS[r['model']]} | {r['aspect_exact']['precision']:.3f} | {r['aspect_exact']['recall']:.3f} | "
            f"{r['pair_f1']['precision']:.3f} | {r['pair_f1']['recall']:.3f} | {r['n_pred']} |"
        )
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    colors = {"lexicon": "#2a78d6", "transformer": "#1f9e74", "llm": "#d0663a"}
    names = [LABELS[r["model"]].split(" (")[0] for r in results]

    # 1) grouped bars of the headline metrics
    metrics = [
        ("Pair F1", lambda r: r["pair_f1"]["f1"]),
        ("Aspect F1\n(exact)", lambda r: r["aspect_exact"]["f1"]),
        ("Aspect F1\n(partial)", lambda r: r["aspect_partial"]["f1"]),
        ("Sentiment\naccuracy", lambda r: r["sentiment_accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        xs = [j + i * width for j in range(len(metrics))]
        vals = [f(r) for _, f in metrics]
        bars = ax.bar(xs, vals, width, label=names[i], color=colors[r["model"]])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks([j + width * (len(results) - 1) / 2 for j in range(len(metrics))], [m for m, _ in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("SemEval-2014 restaurants: accuracy by model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_scores.png", dpi=150)
    plt.close(fig)

    # 2) accuracy vs latency trade-off
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, name in zip(results, names, strict=True):
        ax.scatter(r["latency_ms_mean"], r["pair_f1"]["f1"], s=90, color=colors[r["model"]])
        ax.annotate(
            name, (r["latency_ms_mean"], r["pair_f1"]["f1"]), textcoords="offset points", xytext=(8, 4)
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per sentence (ms, log scale)")
    ax.set_ylabel("Pair F1 (strict)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs. speed")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_tradeoff.png", dpi=150)
    plt.close(fig)

    # 3) sentiment confusion matrices (rows = gold)
    labels = ["positive", "neutral", "negative"]
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.4), squeeze=False)
    for ax, r, name in zip(axes[0], results, names, strict=True):
        m = [[r["confusion"][g][p] for p in labels] for g in labels]
        ax.imshow(m, cmap="Blues")
        for gi in range(3):
            for pi in range(3):
                ax.text(pi, gi, m[gi][pi], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_yticks(range(3), labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_title(name, fontsize=10)
    axes[0][0].set_ylabel("gold")
    fig.suptitle("Sentiment on matched aspects", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_confusion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=ORDER, default=["lexicon"])
    ap.add_argument("--llm-model", default="llama3")
    ap.add_argument("--limit", type=int, help="Only use the first N sentences (smoke test)")
    ap.add_argument("--report-only", action="store_true", help="Only rebuild benchmark.md and charts")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if not args.report_only:
        sentences = load_semeval(download_semeval("restaurants"))[: args.limit]
        for m in args.models:
            print(f"Running {m} on {len(sentences)} sentences ...")
            r = run_model(m, sentences, args.llm_model)
            print(
                f"  pair F1={r['pair_f1']['f1']:.3f}  aspect F1={r['aspect_exact']['f1']:.3f}  "
                f"sentiment acc={r['sentiment_accuracy']:.3f}  {r['latency_ms_mean']:.1f} ms"
            )

    results = load_results()
    if results:
        write_markdown(results)
        write_charts(results)
        print(f"Wrote {RESULTS / 'benchmark.md'} and charts in {IMG}/")


if __name__ == "__main__":
    main()
