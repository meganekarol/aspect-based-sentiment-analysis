import warnings

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import spacy
from spacy.language import Language
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.base import AspectSentiment, ABSAAnalyzer
from src.utils import (
    extract_aspects_spacy,
    NEGATIONS,
    INTENSIFIERS,
    DIMINISHERS,
    squash_confidence,
    choose_label,
)

# -----------------------------------------------------------
# Custom Lexicon Overrides (context fixes for VADER)
# -----------------------------------------------------------
CUSTOM_SENTIMENT = {
    "packed": -0.4, "busy": -0.3, "crowded": -0.5, "slow": -0.4,
    "cold": -0.3, "overpriced": -0.6, "rude": -0.7,
    "friendly": 0.6, "tasty": 0.6, "delicious": 0.7, "fresh": 0.5,
    "perfection": 0.8, "divine": 0.8, "cozy": 0.6, "amazing": 0.7,
    "excellent": 0.7, "wonderful": 0.7, "fantastic": 0.7,
    "terrible": -0.7, "awful": -0.7, "horrible": -0.7, "bland": -0.5
}

# -----------------------------------------------------------
# Common multi-word phrases (scored before token-only logic)
# -----------------------------------------------------------
PHRASE_OVERRIDES = {
    "never disappoints": 0.7,
    "not bad": 0.4,
    "no complaints": 0.4,
    "not good": -0.4,
    "far from good": -0.6,
    "nothing special": -0.3,
}

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
@dataclass
class LexiconABSAConfig:
    """Configuration for the Lexicon-Based ABSA analyzer."""
    model: str = "en_core_web_sm"
    opinion_min_abs: float = 0.10
    pos_th: float = 0.12
    neg_th: float = -0.12


# -----------------------------------------------------------
# Implementation
# -----------------------------------------------------------
class LexiconABSA(ABSAAnalyzer):
    def __init__(self, cfg: Optional[LexiconABSAConfig] = None, nlp: Optional[Language] = None):
        self.cfg = cfg or LexiconABSAConfig()
        self.analyzer = SentimentIntensityAnalyzer()
        self._nlp = nlp or self._load_nlp(self.cfg.model)
        self._phrase_scores: Dict[int, float] = {}

    @staticmethod
    def _load_nlp(model: str) -> Language:
        """
        Load a spaCy model with POS and dependency parsing capabilities.

        Falls back to `spacy.blank("en")` with a sentencizer if the model is
        unavailable, issuing a warning that this limits performance.
        """
        try:
            nlp = spacy.load(model)
            return nlp
        except Exception as e:
            warnings.warn(
                f"[LexiconABSA] Could not load spaCy model '{model}': {e}\n"
                "Falling back to a blank English pipeline (no POS/DEP). "
                "Install 'en_core_web_sm' for full functionality:\n"
                "  python -m spacy download en_core_web_sm",
                UserWarning
            )
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    def _span_sentiment(self,text: str) -> float:
        t = text.lower().strip()
        if t in PHRASE_OVERRIDES:
            return PHRASE_OVERRIDES[t]
        return self.analyzer.polarity_scores(text)["compound"]

    def _extract_aspects(self, doc):
        aspects = {}
        sent_index_by_token = {}
        try:
            for s_i, sent in enumerate(doc.sents):
                for tok in sent:
                    sent_index_by_token[tok.i] = s_i
        except ValueError:
            for tok in doc:
                sent_index_by_token[tok.i] = 0
        base = extract_aspects_spacy(doc)
        for i, (text, span) in base.items():
            aspects[i] = (text, span, sent_index_by_token.get(i, 0))
        return aspects

    def _opinion_score(self, token_text: str) -> float:
        word = token_text.lower()
        if word in CUSTOM_SENTIMENT:
            return CUSTOM_SENTIMENT[word]
        return self.analyzer.polarity_scores(token_text)["compound"]
    # -----------------------------------------------------------
    # Sentiment Modifiers
    # -----------------------------------------------------------
    def _apply_modifiers(self, token, raw_score: float) -> float:
        """
        Adjust sentiment score based on intensifiers, diminishers, and negations.

        Patterns handled:
        - **advmod intensifiers/diminishers** (e.g., "very good", "slightly bad")
        - **negations** attached to the opinion token or its copular head
        - Polarity-sensitive inversion (e.g., "not bad" → mildly positive)
        """
        score = float(raw_score)

        try:
            sent = token.sent
        except Exception:
            sent = token.doc[:]

        # Intensifiers/diminishers via advmod
        factor = 1.0
        for ch in token.children:
            if ch.dep_ == "advmod":
                lt = ch.lower_
                if lt in INTENSIFIERS:
                    factor *= INTENSIFIERS[lt]
                elif lt in DIMINISHERS:
                    factor *= DIMINISHERS[lt]
        score *= factor
        score = max(-1.0, min(1.0, score))

        # Negation handling
        def is_negator(t):
            return t.dep_ == "neg" or t.lower_ in NEGATIONS

        negated = any(is_negator(ch) for ch in token.children)

        # Copular/adjectival complements (e.g., "buffet is fresh")
        if not negated:
            head = token.head
            if head is not None and head.dep_ in {"acomp", "attr"}:
                gov = head.head
                if gov is not None:
                    negated = any(is_negator(ch) for ch in gov.children)

        # Valence-aware negation flip
        if negated:
            if score < 0:
                score = -score * 0.9
            else:
                score = -score * 0.5

        return max(-1.0, min(1.0, score))

    def _is_opinion_verb(self, tok) -> bool:
        return tok.lemma_.lower() in {
            "love", "like", "enjoy", "recommend", "prefer",
            "hate", "dislike", "appreciate", "adore"
        }

    # Build short verb phrase like "never disappoints", "really shines"
    def _build_verb_phrase(self, verb) -> str:
        parts = []
        # include explicit negation and adverbial modifiers closest to the verb
        negs = [c.text for c in verb.children if c.dep_ == "neg"]
        advs = [c.text for c in verb.children if c.dep_ == "advmod"]
        # prefer ordering with negation first (e.g., "never disappoints")
        parts.extend(negs + advs + [verb.text])
        phrase = " ".join(parts) if parts else verb.text
        return phrase

    def _find_aspect_opinions_in_sent(self, sent, aspects_in_sent: Set[int], doc, aspects_idx) -> List[Tuple[int, int]]:
        """
        Identify (aspect, opinion) pairs in a sentence using dependency relations.

        Patterns handled:
        1. **amod** — adjectival modifier ("delicious cake")
        2. **acomp** — adjectival complement ("cake was delicious")
        3. **attr/acomp + nsubj** — copular structures ("risotto was perfection")
        4. **opinion verbs** — ("loved the cake")
        5. **pobj + adjectival ancestor** — ("great for families")
        6. **coordinated adjectives** — ("cozy and beautiful")
        7. **relative clauses** — ("cake which was divine")
        8. **verb with aspect subject** — ("buffet never disappoints")
        """
        pairs = []

        def add(a_i, o_i, phrase: Optional[str] = None):
            if a_i is not None and o_i is not None:
                pairs.append((a_i, o_i))
                if phrase:
                    # store phrase-level sentiment for this opinion token index
                    self._phrase_scores[o_i] = self._span_sentiment(phrase)

        for tok in sent:
            # Pattern 1: amod (adjective modifier) — "delicious cake"
            if tok.dep_ == "amod" and tok.head.i in aspects_in_sent and tok.pos_ == "ADJ":
                add(tok.head.i, tok.i)

            # Pattern 2: acomp — "cake was delicious"
            elif tok.dep_ == "acomp" and tok.pos_ == "ADJ":
                for c in tok.head.children:
                    if c.dep_ in {"nsubj", "nsubjpass"} and c.i in aspects_in_sent:
                        add(c.i, tok.i)

            # Pattern 3: nsubj + attr/acomp — "risotto was perfection"
            elif tok.dep_ in {"nsubj", "nsubjpass"} and tok.i in aspects_in_sent:
                for ch in tok.head.children:
                    if ch.dep_ in {"acomp", "attr"} and ch.pos_ in {"ADJ", "NOUN"}:
                        add(tok.i, ch.i)

            # Pattern 4: opinion verb dobj — "loved the cake"
            elif tok.dep_ == "dobj" and tok.i in aspects_in_sent:
                if tok.head.pos_ == "VERB" and self._is_opinion_verb(tok.head):
                    add(tok.i, tok.head.i)

            # Pattern 5: pobj with adjectival ancestor — "great for families"
            elif tok.dep_ == "pobj" and tok.i in aspects_in_sent:
                for anc in tok.ancestors:
                    if anc.pos_ == "ADJ":
                        add(tok.i, anc.i)
                        break

            # Pattern 6: coordinated adjectives — "cozy and beautiful"
            elif tok.dep_ == "conj" and tok.pos_ == "ADJ":
                if tok.head.dep_ == "amod" and tok.head.head.i in aspects_in_sent:
                    add(tok.head.head.i, tok.i)
                elif tok.head.dep_ == "acomp":
                    for subj in tok.head.head.children:
                        if subj.dep_ in {"nsubj", "nsubjpass"} and subj.i in aspects_in_sent:
                            add(subj.i, tok.i)

            # Pattern 7: relative clause — "cake which was divine"
            elif tok.dep_ == "acomp" and tok.head.dep_ == "relcl":
                for anc in tok.head.ancestors:
                    if anc.pos_ in {"NOUN", "PROPN"} and anc.i in aspects_in_sent:
                        add(anc.i, tok.i)
                        break

            # Pattern 8: verb with subject aspect — "buffet never disappoints"
            if tok.pos_ == "VERB":
                subj = next((c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"} and c.i in aspects_in_sent), None)
                if subj is not None:
                    phrase = self._build_verb_phrase(tok)  # e.g., "never disappoints"
                    add(subj.i, tok.i, phrase=phrase)

        return pairs

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self._nlp(text)
        aspects_idx = self._extract_aspects(doc)

        # reset per-call phrase store
        self._phrase_scores = {}

        by_sent: Dict[int, Set[int]] = {}
        for a_i, (_, _, s_id) in aspects_idx.items():
            by_sent.setdefault(s_id, set()).add(a_i)

        agg_by_aspect: Dict[str, float] = {}
        hits_by_aspect: Dict[str, int] = {}

        try:
            sentences = list(doc.sents)
        except ValueError:
            sentences = [doc]

        for s_id, sent in enumerate(sentences):
            aspects_in_sent = by_sent.get(s_id, set())
            if not aspects_in_sent:
                continue

            if doc.has_annotation("DEP"):
                pairs = self._find_aspect_opinions_in_sent(sent, aspects_in_sent, doc, aspects_idx)
            else:
                pairs = []

            for a_i, o_i in pairs:
                a_text = aspects_idx[a_i][0]
                op_tok = doc[o_i]

                # prefer phrase-level sentiment if we built one (e.g., "never disappoints")
                if o_i in self._phrase_scores:
                    raw = self._phrase_scores[o_i]
                else:
                    raw = self._opinion_score(op_tok.text)

                if abs(raw) >= self.cfg.opinion_min_abs:
                    score = self._apply_modifiers(op_tok, raw)
                    agg_by_aspect[a_text] = agg_by_aspect.get(a_text, 0.0) + score
                    hits_by_aspect[a_text] = hits_by_aspect.get(a_text, 0) + 1

        results = []
        for a_i, (a_text, span, _) in aspects_idx.items():
            total = agg_by_aspect.get(a_text, 0.0)
            count = hits_by_aspect.get(a_text, 0)
            if count == 0:
                continue

            avg = total / count
            label = choose_label(avg, self.cfg.pos_th, self.cfg.neg_th)
            conf = squash_confidence(avg)

            results.append(
                AspectSentiment(
                    aspect=a_text,
                    sentiment=label,
                    confidence=conf,
                    text_span=span
                )
            )
        return results

if __name__ == "__main__":

    model = LexiconABSA(LexiconABSAConfig())
    text = "The food was great but the service was slow."
    results = model.analyze(text)

    for r in results:
        print(f"   ➜ {r.aspect:<20} | {r.sentiment:<10} | conf={r.confidence:.2f}")
