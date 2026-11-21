"""Evaluation of ABSA predictions against gold aspect-term annotations.

Metrics (micro-averaged over the corpus):

- **Aspect extraction P/R/F1**, with two matching modes:
  ``exact`` (normalized strings equal) and ``partial`` (share a content word).
- **Sentiment accuracy / macro-F1** on aspects that were found (partial match),
  i.e. "given the aspect was found, was its polarity right?".
- **Pair F1** (end-to-end): an aspect counts only if it is an exact match
  *and* its polarity is correct. This is the strictest, headline metric.

Gold terms labelled ``conflict`` are excluded, as is standard for
SemEval-2014: predictions matched to them are ignored rather than counted
as false positives.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from absa.base import SENTIMENTS, AspectSentiment
from absa.datasets import GoldSentence

_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "our", "my", "their", "his", "her", "its"}
_STOPWORDS = _DETERMINERS | {"of", "and", "for", "with", "to", "in", "on", "at", "'s"}
_PUNCT = re.compile(r"[^\w\s'&-]")


def normalize(term: str) -> str:
    """Lowercase, strip punctuation and leading determiners: 'The Service!' -> 'service'."""
    words = _PUNCT.sub(" ", term.lower()).split()
    while words and words[0] in _DETERMINERS:
        words = words[1:]
    return " ".join(words)


def _content_words(term: str) -> set[str]:
    return {w for w in normalize(term).split() if w not in _STOPWORDS}


def match_aspects(gold: list[str], pred: list[str]) -> list[tuple[int, int, str]]:
    """Greedy one-to-one matching; returns ``(gold_idx, pred_idx, 'exact'|'partial')``."""
    pairs: list[tuple[int, int, str]] = []
    free_g, free_p = set(range(len(gold))), set(range(len(pred)))
    for gi in sorted(free_g):
        for pi in sorted(free_p):
            if normalize(gold[gi]) == normalize(pred[pi]):
                pairs.append((gi, pi, "exact"))
                free_g.discard(gi)
                free_p.discard(pi)
                break
    for gi in sorted(free_g):
        g_words = _content_words(gold[gi])
        best, best_overlap = None, 0
        for pi in sorted(free_p):
            overlap = len(g_words & _content_words(pred[pi]))
            if overlap > best_overlap:
                best, best_overlap = pi, overlap
        if best is not None:
            pairs.append((gi, best, "partial"))
            free_p.discard(best)
    return pairs


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


@dataclass
class EvalResult:
    n_sentences: int
    n_gold: int
    n_pred: int
    aspect_exact: dict[str, float]
    aspect_partial: dict[str, float]
    sentiment_accuracy: float
    sentiment_macro_f1: float
    pair_f1: dict[str, float]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)  # gold -> pred -> count

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(gold: list[GoldSentence], preds: list[list[AspectSentiment]]) -> EvalResult:
    """Score ``preds`` (one list per gold sentence, same order) against ``gold``."""
    if len(gold) != len(preds):
        raise ValueError(f"{len(gold)} gold sentences but {len(preds)} prediction lists")

    n_gold = n_pred = tp_exact = tp_partial = tp_pair = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for sent, pred in zip(gold, preds, strict=True):
        g_terms = [a.term for a in sent.aspects]
        p_terms = [p.aspect for p in pred]
        pairs = match_aspects(g_terms, p_terms)

        conflict_g = {i for i, a in enumerate(sent.aspects) if a.polarity == "conflict"}
        ignored_p = {pi for gi, pi, _ in pairs if gi in conflict_g}
        n_gold += len(sent.aspects) - len(conflict_g)
        n_pred += len(pred) - len(ignored_p)

        for gi, pi, kind in pairs:
            if gi in conflict_g:
                continue
            g_pol, p_pol = sent.aspects[gi].polarity, pred[pi].sentiment
            tp_partial += 1
            confusion[(g_pol, p_pol)] += 1
            if kind == "exact":
                tp_exact += 1
                tp_pair += g_pol == p_pol

    matched = sum(confusion.values())
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    per_class_f1 = []
    for label in SENTIMENTS:
        tp = confusion[(label, label)]
        n_p = sum(v for (_, p), v in confusion.items() if p == label)
        n_g = sum(v for (g, _), v in confusion.items() if g == label)
        per_class_f1.append(_prf(tp, n_p, n_g)["f1"])

    return EvalResult(
        n_sentences=len(gold),
        n_gold=n_gold,
        n_pred=n_pred,
        aspect_exact=_prf(tp_exact, n_pred, n_gold),
        aspect_partial=_prf(tp_partial, n_pred, n_gold),
        sentiment_accuracy=round(correct / matched, 4) if matched else 0.0,
        sentiment_macro_f1=round(sum(per_class_f1) / len(per_class_f1), 4),
        pair_f1=_prf(tp_pair, n_pred, n_gold),
        confusion={g: {p: confusion[(g, p)] for p in SENTIMENTS} for g in SENTIMENTS},
    )
