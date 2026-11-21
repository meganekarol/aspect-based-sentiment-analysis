# Error analysis

Where and why each approach fails on the SemEval-2014 restaurant test set. The
examples come from `results/predictions/<model>.jsonl`, written by
`scripts/benchmark.py`, so every case here can be reproduced.

## Lexicon (spaCy + VADER)

Out of 1,120 gold aspects, the lexicon model misses about 690. It has high
precision on polarity (93% accuracy once an aspect is found) but low recall. The
errors fall into four groups:

| Error type | Count | Example | Gold | Predicted |
|---|---|---|---|---|
| **Missed: no opinion link** | 504 | "I have to say they have one of the *fastest* **delivery times** in the city." | delivery times: positive | – |
| **Missed: neutral aspect** | 182 | "Entrees include classics like **lasagna**, fettuccine Alfredo and chicken parmigiana." | lasagna: neutral | – |
| **False positive** | 143 | "The bread is top **notch** as well." | (bread) | notch: positive |
| **Wrong polarity** | 29 | "The **staff** should be a bit more friendly." | staff: negative | staff: positive |

**What's behind it**

1. **Only aspects with an opinion are reported.** An aspect is output only if a
   dependency pattern connects it to a sentiment word. Superlatives inside a
   noun phrase ("one of the fastest delivery times"), sentence fragments
   ("BEST spicy tuna roll") and verbs outside the opinion-verb list are missed.
2. **It never predicts neutral.** Neutral aspects carry no opinion word, so they
   are always missed: 16% of the gold aspects. Reporting every opinion-less noun
   as neutral (`LexiconABSAConfig(emit_neutral=True)`) was tried and **lowered**
   pair F1 from 0.417 to 0.312. Recall rises, but most nouns in a review are not
   annotated aspects, so precision collapses. The option is off by default.
3. **Polarity beyond the word level.** Modal and counterfactual sentences ("should
   be *more* friendly") and pragmatic negatives ("a shame that he didn't have a
   supportive staff") get the polarity of the adjective, not of the sentence.
4. **Parser errors.** `en_core_web_sm` tags "tikka" in "chicken tikka masala"
   as a VERB and "notch" in "top notch" as the head noun. A larger spaCy model
   (`en_core_web_trf`) would likely help, at a large speed cost.

The two known-limitation unit tests (`test_compound_aspect`, `test_sarcasm`)
are kept as `xfail` so the behaviour is documented and any improvement shows up.

## Transformer (DeBERTa-v3)

The transformer finds 81% of gold aspects (exact match), so its errors are
more about boundaries and polarity than missed aspects.

| Error type | Count | Example | Gold | Predicted |
|---|---|---|---|---|
| **False positive** | 211 | "I absolutely suggest this **place**." | (none) | place: positive |
| **Wrong polarity** | 153 | "Sandwiches, **burgers** and salads … are classic successes." | burgers: positive | burgers: neutral |
| **Missed** | 137 | "Desserts are almost incredible: my personal favorite is their **Tart of the Day**." | Tart of the Day: positive | – |
| **Boundary (partial match)** | 79 | "The **spicy mussels** are a highlight." | spicy mussels | mussels |

**What's behind it**

1. **Annotation convention, not model error.** About 40 of the false positives
   are *place* (25) and *restaurant* (15). SemEval treats the restaurant as the
   review's target and does not annotate it as an aspect term. Filtering these
   words would raise the score, but it would be tuning to the test set, so
   it was not done.
2. **Over-predicting neutral.** 79 of the 153 polarity errors are positive
   aspects predicted *neutral*, typically in lists ("sandwiches, burgers and
   salads … are successes") where the opinion is far from the aspect.
3. **Boundary disagreements.** Modifiers are included or dropped inconsistently
   ("mussels" vs. gold "spicy mussels", "serious indian food" vs. "indian food").
   Part of this is annotation ambiguity. Strict pair F1 counts these as wrong;
   the partial-match aspect F1 (0.854) shows the aspect was found.
4. **Fixed along the way.** Sub-word pieces had split one aspect into fragments
   ("outdoor" + "seating"), cut words short ("staff memeber") and shifted spans
   by one character. Fixing these raised pair F1 from 0.647 to 0.681. See
   [design.md](design.md#fixes-found-by-benchmarking).

## LLM (Ollama)

The LLM back-end is benchmarked locally (it needs an Ollama server):
`python scripts/benchmark.py --models llm --llm-model llama3`. Its predictions
are then in `results/predictions/llm.jsonl`.


## Takeaways

- The transformer is the strongest model overall (pair F1 0.681 vs. 0.417), at
  about 40x the latency of the lexicon model on CPU.
- The rule-based model is precise and explainable but limited by its recall.
  Its errors can be traced to a specific rule, which is useful when you need to
  justify a prediction.
- Benchmarking found real bugs that the hand-written unit tests had missed:
  negation handling, aspect boundaries and span offsets. See
  [design.md](design.md#fixes-found-by-benchmarking).
