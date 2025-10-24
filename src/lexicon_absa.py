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
# Configuration
# -----------------------------------------------------------
@dataclass
class LexiconABSAConfig:
    model: str = "en_core_web_sm"
    proximity_radius: int = 0          # 0 = disabled; only used if no DEP
    opinion_min_abs: float = 0.08      # lower threshold to catch subtler opinions
    pos_th: float = 0.05
    neg_th: float = -0.05
    overall_aspect: Optional[str] = "overall"  # optional fallback aspect


# -----------------------------------------------------------
# Implementation
# -----------------------------------------------------------
class LexiconABSA(ABSAAnalyzer):
    def __init__(self, cfg: Optional[LexiconABSAConfig] = None, nlp: Optional[Language] = None):
        self.cfg = cfg or LexiconABSAConfig()
        self.analyzer = SentimentIntensityAnalyzer()
        self._nlp = nlp or self._load_nlp(self.cfg.model)

    @staticmethod
    def _load_nlp(model: str) -> Language:
        try:
            return spacy.load(model)
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    def _extract_aspects(self, doc):
        """Extract aspects with sentence IDs"""
        aspects = {}
        sent_index_by_token = {}

        try:
            for s_i, sent in enumerate(doc.sents):
                for tok in sent:
                    sent_index_by_token[tok.i] = s_i
        except ValueError:
            # No sentence segmentation
            for tok in doc:
                sent_index_by_token[tok.i] = 0

        base = extract_aspects_spacy(doc)
        for i, (text, span) in base.items():
            aspects[i] = (text, span, sent_index_by_token.get(i, 0))

        return aspects

    def _opinion_score(self, token_text: str) -> float:
        """Get opinion score with custom overrides"""
        word = token_text.lower()
        if word in CUSTOM_SENTIMENT:
            return CUSTOM_SENTIMENT[word]
        return self.analyzer.polarity_scores(token_text)["compound"]

    def _apply_modifiers(self, token, raw_score: float) -> float:
        """Apply intensity modifiers and negation"""
        score = raw_score
        doc = token.doc
        window = 3
        left_i = max(0, token.i - window)
        right_i = min(len(doc), token.i + window + 1)
        neighbors = [t for t in doc[left_i:right_i] if t.i != token.i]

        # Apply intensifiers/diminishers first
        factor = 1.0
        for t in neighbors + list(token.children):
            lt = t.lower_
            if lt in INTENSIFIERS:
                factor *= INTENSIFIERS[lt]
            elif lt in DIMINISHERS:
                factor *= DIMINISHERS[lt]

        score *= factor
        score = max(-1.0, min(1.0, score))

        # Check for negation
        def is_negator(t):
            return t.lower_ in NEGATIONS or t.dep_ == "neg"

        negated = any(is_negator(ch) for ch in token.children)
        if not negated:
            negated = any(is_negator(t) for t in neighbors)
        if not negated and token.head and token.head.dep_ in {"cop", "aux", "auxpass"}:
            negated = any(is_negator(ch) for ch in token.head.children)

        if negated:
            score *= -1

        return score

    def _is_opinion_verb(self, tok) -> bool:
        """Check if verb expresses opinion"""
        return tok.lemma_.lower() in {
            "love", "like", "enjoy", "recommend", "prefer",
            "hate", "dislike", "appreciate", "adore"
        }

    def _find_aspect_opinions_in_sent(
            self, sent, aspects_in_sent: Set[int], doc, aspects_idx
    ) -> List[Tuple[int, int]]:
        """Find (aspect_idx, opinion_idx) pairs in sentence"""
        pairs = []

        def add(a_i, o_i):
            if a_i is not None and o_i is not None:
                pairs.append((a_i, o_i))

        for tok in sent:
            # Pattern 1: amod (adjective modifier)
            # "delicious cake"
            if tok.dep_ == "amod" and tok.head.i in aspects_in_sent:
                add(tok.head.i, tok.i)

            # Pattern 2: acomp (adjectival complement)
            # "cake was delicious"
            elif tok.dep_ == "acomp" and tok.pos_ == "ADJ":
                for c in tok.head.children:
                    if c.dep_ in {"nsubj", "nsubjpass"} and c.i in aspects_in_sent:
                        add(c.i, tok.i)

            # Pattern 3: nsubj with adjectival complement
            # "risotto was perfection"
            elif tok.dep_ in {"nsubj", "nsubjpass"} and tok.i in aspects_in_sent:
                for ch in tok.head.children:
                    if ch.dep_ in {"acomp", "attr"} and ch.pos_ in {"ADJ", "NOUN"}:
                        add(tok.i, ch.i)

            # Pattern 4: Opinion verb with direct object
            # "loved the cake"
            elif tok.dep_ == "dobj" and tok.i in aspects_in_sent:
                if tok.head.pos_ == "VERB" and self._is_opinion_verb(tok.head):
                    add(tok.i, tok.head.i)

            # Pattern 5: Prepositional object
            # "great for families"
            elif tok.dep_ == "pobj" and tok.i in aspects_in_sent:
                for anc in tok.ancestors:
                    if anc.pos_ == "ADJ":
                        add(tok.i, anc.i)
                        break

            # Pattern 6: Coordinated adjectives
            # "cozy and beautiful"
            elif tok.dep_ == "conj" and tok.pos_ == "ADJ":
                if tok.head.dep_ in {"acomp", "amod"}:
                    # Find the subject
                    if tok.head.dep_ == "amod" and tok.head.head.i in aspects_in_sent:
                        add(tok.head.head.i, tok.i)
                    elif tok.head.dep_ == "acomp":
                        for subj in tok.head.head.children:
                            if subj.dep_ in {"nsubj", "nsubjpass"} and subj.i in aspects_in_sent:
                                add(subj.i, tok.i)

            # Pattern 7: Relative clause
            # "cake which was divine"
            elif tok.dep_ == "acomp" and tok.head.dep_ == "relcl":
                for anc in tok.head.ancestors:
                    if anc.pos_ in {"NOUN", "PROPN"} and anc.i in aspects_in_sent:
                        add(anc.i, tok.i)
                        break

        return pairs

    def analyze(self, text: str) -> List[AspectSentiment]:
        """Main analysis method"""
        doc = self._nlp(text)
        aspects_idx = self._extract_aspects(doc)

        # Group aspects by sentence
        by_sent: Dict[int, Set[int]] = {}
        for a_i, (_, _, s_id) in aspects_idx.items():
            by_sent.setdefault(s_id, set()).add(a_i)

        # Aggregate scores
        agg_by_aspect: Dict[str, float] = {}
        hits_by_aspect: Dict[str, int] = {}

        # Process each sentence
        try:
            sentences = list(doc.sents)
        except ValueError:
            sentences = [doc]

        for s_id, sent in enumerate(sentences):
            aspects_in_sent = by_sent.get(s_id, set())

            if not aspects_in_sent:
                continue

            # Find opinion-aspect pairs
            if doc.has_annotation("DEP"):
                pairs = self._find_aspect_opinions_in_sent(sent, aspects_in_sent, doc, aspects_idx)
            else:
                pairs = []

            # Score each pair
            for a_i, o_i in pairs:
                a_text = aspects_idx[a_i][0]
                op_tok = doc[o_i]

                raw = self._opinion_score(op_tok.text)

                # Only process if opinion is strong enough
                if abs(raw) >= self.cfg.opinion_min_abs:
                    score = self._apply_modifiers(op_tok, raw)
                    agg_by_aspect[a_text] = agg_by_aspect.get(a_text, 0.0) + score
                    hits_by_aspect[a_text] = hits_by_aspect.get(a_text, 0) + 1

        # Build results
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
    import pandas as pd

    analyzer = LexiconABSA(LexiconABSAConfig(overall_aspect=None))

    reviews = [
  "Absolutely incredible experience! The staff went above and beyond to make our anniversary dinner special. The mushroom risotto was perfection, and don't even get me started on the chocolate lava cake. Worth every penny!",
  "Really wanted to love this place based on all the hype, but left disappointed. Service was slow even though it wasn't busy, and my steak was overcooked despite ordering medium-rare. The ambiance was nice though.",
  "Great spot for brunch! The avocado toast was fresh and flavorful. Only complaint is that it gets pretty noisy when it's crowded, making conversation difficult. Would definitely come back on a weekday.",
  "Worst dining experience in years. Found a hair in my soup and when I told the server, they barely apologized. Manager never even came to our table. Will not be returning.",
  "Hidden gem! This little family-owned restaurant has the most authentic cuisine I've tasted outside of Italy. The owner came out and chatted with us about the recipes. Felt like eating at a friend's house. Can't recommend enough!",
  "It's okay. Nothing special but nothing terrible either. Prices are reasonable for the portion sizes. Good option if you're in the area and hungry, but I wouldn't go out of my way to come here.",
  "Love the vibe here! Super cozy atmosphere with dim lighting and great music. The cocktails are creative and strong. Food was solid - the Brussels sprouts appetizer was a standout. Lost a star because wait times can be long on weekends.",
  "Best pizza in town, hands down. The crust is perfectly crispy, and they don't skimp on toppings. Been coming here for three years and it's consistently excellent. The garlic knots are also addictive!",
  "Overpriced for what you get. My salad was $18 and mostly iceberg lettuce with a few toppings. The server seemed annoyed when we asked questions about the menu. Expected better based on the reviews.",
  "Solid neighborhood spot with a great happy hour! Wings are crispy and saucy, and the beer selection is impressive. Can get crowded during games, but that's part of the fun. Parking can be tricky though."
]

    for i, review in enumerate(reviews, 1):
        print(f"\nReview {i}: {review}\n")
        aspects = analyzer.analyze(review)

        if not aspects:
            print("  No clear aspects detected.")
        else:
            for asp in aspects:
                print(f"   ➜ {asp.aspect:<25} | {asp.sentiment:<10} | conf={asp.confidence:.2f}")

    # analyzer = LexiconABSA(LexiconABSAConfig(overall_aspect=None))
    # text = """Oh perfect, another burnt pizza. Amazing job."""
    # results = analyzer.analyze(text)
    # for r in results:
    #     print(f"{r.aspect:<20} | {r.sentiment:<8} | conf={r.confidence:.2f}")