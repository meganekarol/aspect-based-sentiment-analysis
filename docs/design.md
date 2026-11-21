# Design decisions

How each analyzer works and why it is built that way. For results see [`results/benchmark.md`](../results/benchmark.md); for failure cases see [error_analysis.md](error_analysis.md).

## LexiconABSA (Rule-Based)

- **Use of Noun Chunks:**  
  We used spaCy’s `noun_chunks` as aspect candidates because they capture complete, meaningful entities (e.g., “mushroom risotto”, “customer service”) rather than isolated words.

- **Removal of Pronouns:**  
  Pronouns like *it*, *they*, and *this* are context-dependent and not meaningful aspects on their own. Filtering them prevents false positives.

- **Sentence-Level Analysis:**  
  Sentiment expressions are typically contained within a sentence. Processing each sentence separately avoids sentiment spillover between clauses.

- **Custom Sentiment Lexicon & Phrase Overrides:**  
  To improve domain coverage, custom entries (e.g., “cozy”, “divine”, “worth every penny”) and idiomatic overrides (e.g., “not bad”, “never disappoints”) were added. This ensures polarity alignment in common restaurant-review contexts.

- **Aspect–Opinion Linking Patterns:**  
  Implemented in `_find_aspect_opinions_in_sent()`, this function uses dependency relations like:
  - `amod` / `acomp` → adjectival modifiers (“delicious pizza”)  
  - `nsubj` + copular verbs → (“service was slow”)  
  - `verb-object` pairs → (“staff disappointed”)  
  - Coordinations and relative clauses → (“cake, which was divine”)  
  These patterns robustly map aspects to their sentiment expressions in varied syntax.

---

## TransformerABSA (Hybrid Pretrained Models)

- **Hybridization Decision:**  
  During model exploration:
  - `yangheng/deberta-v3-base-end2end-absa` was strong at **aspect extraction** but weaker in sentiment labeling.  
  - `yangheng/deberta-v3-base-absa-v1.1` excelled in **sentiment classification** but did not extract aspects automatically.  
  The solution was to **hybridize both models** one to detect aspects, the other to classify their sentiment achieving higher precision and recall together.

- **Exception-Safe Loading:**  
  Implemented try–except logic for model initialization to handle missing or incompatible Hugging Face models gracefully.

- **Confidence-Based Merging:**  
  If a sentiment prediction has low confidence but aspect extraction is strong, the model uses aspect confidence as a fallback for stability.

---

## LLMABSA (Ollama – Prompt-Engineered LLM)

- **Prompt Engineering:**  
  A structured system prompt was designed to enforce JSON output, guide the model to identify noun-based aspects, and classify each as *positive*, *negative*, or *neutral*.  
  Few-shot examples were added to stabilize responses and reduce ambiguity.

- **Strict Output Schema:**  
  The model is instructed to return responses as:
  ```json
  [
    {"aspect": "service", "sentiment": "negative", "confidence": 0.85}
  ]
  ```

---

## Fixes found by benchmarking

Measuring against SemEval-2014 gold labels exposed bugs that the unit tests
had not caught:

| Analyzer | Problem | Fix | Effect |
|---|---|---|---|
| Lexicon | Negation was ignored in copular sentences. In "the food was **not** good" the negator attaches to *was*, but the code looked for it on the head of *good*'s head. "not good" scored positive, "not bad" negative. | Check the copula's children when the opinion word is an `acomp`/`attr` (`_apply_modifiers`). | 2 previously failing unit tests now pass |
| Lexicon | Opinion adjectives were folded into the aspect name ("**fresh** sushi", "**friendly** staff"), so exact matches with gold terms failed. | `extract_aspects_spacy(is_opinion=...)` keeps sentiment-bearing adjectives out of the name; neutral ones ("*live* entertainment") stay. Spans now cover only the aspect tokens. | Pair F1 0.256 → 0.417 |
| Transformer | Spans included the SentencePiece leading space (`(3, 20)` = " mushroom risotto"). | Trim whitespace from offsets. | Pair F1 0.647 → 0.671 (with the merge fix below) |
| Transformer | One aspect was split into adjacent pieces ("outdoor" + "seating") when its tokens carried different polarity tags. | Merge aspect groups separated only by whitespace. | (included above) |
| Transformer | Sub-word groups could stop mid-word ("staff memeber" for "memebers", "Veal Chop"). | Extend spans to word boundaries. | Pair F1 0.671 → 0.681 |
| LLM | Unparseable model output returned `[]` without retrying. Invalid labels and out-of-range confidences were passed through. | Parse failures raise and trigger a retry; labels are validated and confidence clamped to [0, 1]. | Covered by new unit tests |

