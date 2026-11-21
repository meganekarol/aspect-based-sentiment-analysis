# Benchmark: SemEval-2014 Task 4, restaurants (gold test set)

800 sentences, 1120 gold aspect terms (14 `conflict` terms excluded).

| Model | Pair F1 (strict) | Aspect F1 (exact) | Aspect F1 (partial) | Sentiment acc. | Sentiment macro-F1 | Latency (ms/sentence) |
|---|---|---|---|---|---|---|
| Lexicon (spaCy + VADER) | **0.417** | 0.445 | 0.514 | 0.933 | 0.609 | 9 |
| Transformer (DeBERTa-v3) | **0.681** | 0.785 | 0.854 | 0.844 | 0.794 | 386 |

- **Pair F1 (strict)**: aspect string matches exactly *and* polarity is correct.
- **Aspect F1**: aspect extraction only; *partial* = shares a content word with the gold term.
- **Sentiment acc. / macro-F1**: polarity on aspects that were found (partial match).
- Latency: mean per sentence on AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD (CPU), after warm-up.

Precision / recall detail:

| Model | Aspect P (exact) | Aspect R (exact) | Pair P | Pair R | Predicted aspects |
|---|---|---|---|---|---|
| Lexicon (spaCy + VADER) | 0.661 | 0.335 | 0.621 | 0.314 | 567 |
| Transformer (DeBERTa-v3) | 0.765 | 0.807 | 0.663 | 0.700 | 1182 |
