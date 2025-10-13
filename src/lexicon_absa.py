# src/lexicon_absa.py
import os
import json
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .base import ABSAAnalyzer, AspectSentiment

class LexiconABSA(ABSAAnalyzer):
    """
    Lexicon-Based Aspect-Based Sentiment Analysis (ABSA) implementation.
    Combines spaCy dependency parsing and VADER sentiment scoring.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.sid = SentimentIntensityAnalyzer()

        # Load domain-specific Yelp lexicon if available
        domain_path = os.path.join(os.path.dirname(__file__), "../data/yelp_domain_lexicon.json")
        if os.path.exists(domain_path):
            with open(domain_path, "r", encoding="utf-8") as f:
                domain_lex = json.load(f)
                for word, score in domain_lex.items():
                    # scale [-1, 1] → VADER’s internal scale (~-4 to +4)
                    self.sid.lexicon[word] = score * 4.0

    # ------------------------------------------------------------
    # ASPECT-OPINION EXTRACTION PIPELINE
    # ------------------------------------------------------------

    def _extract_aspects(self, doc):
        """Extract noun-based aspects (exclude pronouns)."""
        return [
            chunk.root.text.lower()
            for chunk in doc.noun_chunks
            if chunk.root.pos_ != "PRON"
        ]

    def _find_opinions(self, doc, aspects):
        """Find opinion-bearing words (adjectives/verbs) connected to aspects."""
        opinions = []
        for token in doc:
            # adjective modifying noun (e.g., "delicious food")
            if token.dep_ == "amod" and token.head.text.lower() in aspects:
                opinions.append((token.head.text.lower(), token.text.lower(), token))

            # adjective complement (e.g., "food was delicious")
            elif token.dep_ == "acomp":
                for child in token.head.children:
                    if child.dep_ == "nsubj" and child.text.lower() in aspects:
                        opinions.append((child.text.lower(), token.text.lower(), token))

            # adverb modifying adjective/verb (e.g., "bit high")
            elif token.dep_ == "advmod" and token.head.pos_ in ["ADJ", "VERB"]:
                for child in token.head.children:
                    if child.dep_ == "nsubj" and child.text.lower() in aspects:
                        opinions.append((child.text.lower(), token.text.lower(), token))
        return opinions

    def _score_opinions(self, opinions):
        """Compute sentiment scores for aspects based on attached opinions."""
        aspect_sentiments = {}

        for aspect, opinion_word, token in opinions:
            base_score = self.sid.polarity_scores(opinion_word)["compound"]

            # --- Negation ---
            neg_factor = -1 if any(child.dep_ == "neg" for child in token.children) else 1
            if token.head:
                neg_factor *= -1 if any(child.dep_ == "neg" for child in token.head.children) else 1

            # --- Intensifiers ---
            intensity_factor = 1.0
            for child in token.children:
                if child.dep_ == "advmod" and child.text.lower() in ["very", "extremely", "highly", "so", "really"]:
                    intensity_factor += 0.5
            if token.head:
                for child in token.head.children:
                    if child.dep_ == "advmod" and child.text.lower() in ["very", "extremely", "highly", "so", "really"]:
                        intensity_factor += 0.5

            # Adjusted score
            final_score = base_score * neg_factor * intensity_factor
            if aspect not in aspect_sentiments:
                aspect_sentiments[aspect] = []
            aspect_sentiments[aspect].append(final_score)

        # Aggregate
        for aspect, scores in aspect_sentiments.items():
            avg = sum(scores) / len(scores)
            sentiment = "positive" if avg >= 0.05 else "negative" if avg <= -0.05 else "neutral"
            aspect_sentiments[aspect] = (sentiment, avg)

        return aspect_sentiments

    # ------------------------------------------------------------
    # MAIN INTERFACE METHOD
    # ------------------------------------------------------------

    def analyze(self, text: str):
        """
        Analyze a review and return a list of AspectSentiment objects.
        """
        doc = self.nlp(text)
        aspects = self._extract_aspects(doc)
        opinions = self._find_opinions(doc, aspects)
        aspect_scores = self._score_opinions(opinions)

        results = []
        for aspect, (sentiment, avg_score) in aspect_scores.items():
            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=round(abs(avg_score), 3),
                )
            )

        return results