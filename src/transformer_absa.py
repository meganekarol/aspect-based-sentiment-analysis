"""
TransformerABSA — Hybrid Transformer-Based ABSA Implementation
--------------------------------------------------------------
Combines:
- Aspect extraction using: yangheng/deberta-v3-base-end2end-absa
- Aspect-conditioned sentiment classification using: yangheng/deberta-v3-base-absa-v1.1

Key Features:
- Exception-safe model loading
- Clear docstrings for maintainability
- Consistent API with LexiconABSA and other analyzers
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

from src.base import ABSAAnalyzer, AspectSentiment


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
@dataclass
class TransformerABSAConfig:
    """Configuration for the Transformer-based ABSA implementation."""
    aspect_model: str = "yangheng/deberta-v3-base-end2end-absa"
    sentiment_model: str = "yangheng/deberta-v3-base-absa-v1.1"
    sentiment_conf_threshold: float = 0.6
    aggregation_strategy: str = "simple"
    device: int = 0 if torch.cuda.is_available() else -1


# ------------------------------------------------------------
# Implementation
# ------------------------------------------------------------
class TransformerABSA(ABSAAnalyzer):
    """
    Transformer-based Aspect-Based Sentiment Analyzer.
    Extracts aspect terms using a token-classification model, and
    classifies sentiment for each aspect using a separate sentiment model.
    """

    def __init__(self, cfg: Optional[TransformerABSAConfig] = None):
        self.cfg = cfg or TransformerABSAConfig()

        # Load models safely
        self.aspect_pipeline = self._safe_load_aspect_pipeline()
        self.sentiment_tokenizer, self.sentiment_model = self._safe_load_sentiment_model()

        # Sentiment label mapping (for DeBERTa ABSA)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    # --------------------------------------------------------
    # Safe model loading
    # --------------------------------------------------------
    def _safe_load_aspect_pipeline(self):
        """Safely loads the aspect extraction pipeline."""
        try:
            pipe = pipeline(
                "token-classification",
                model=self.cfg.aspect_model,
                aggregation_strategy=self.cfg.aggregation_strategy,
                device=self.cfg.device,
            )
            return pipe
        except Exception as e:
            warnings.warn(
                f"[TransformerABSA] Failed to load aspect extraction model "
                f"('{self.cfg.aspect_model}'): {e}\n"
                "Ensure the model is available or check your internet connection.\n"
                "To install manually:\n"
                f"  from transformers import AutoModelForTokenClassification\n"
                f"  AutoModelForTokenClassification.from_pretrained('{self.cfg.aspect_model}')",
                UserWarning,
            )
            return None

    def _safe_load_sentiment_model(self):
        """Safely loads the sentiment classification model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.sentiment_model)
            model = AutoModelForSequenceClassification.from_pretrained(self.cfg.sentiment_model)
            if self.cfg.device >= 0:
                model.to(self.cfg.device)
            return tokenizer, model
        except Exception as e:
            warnings.warn(
                f"[TransformerABSA] Failed to load sentiment model "
                f"('{self.cfg.sentiment_model}'): {e}\n"
                "Ensure the model is downloaded and accessible.\n"
                "Try:\n"
                f"  from transformers import AutoModelForSequenceClassification\n"
                f"  AutoModelForSequenceClassification.from_pretrained('{self.cfg.sentiment_model}')",
                UserWarning,
            )
            return None, None

    # --------------------------------------------------------
    # Aspect Extraction
    # --------------------------------------------------------
    def _extract_aspects(self, text: str) -> List[Tuple[str, Tuple[int, int], float]]:
        """
        Extract aspect terms from text using a token-classification model.

        Args:
            text (str): Input review text.

        Returns:
            List of tuples (aspect_text, (start_char, end_char), confidence)
            Example: [("service", (12, 19), 0.95)]
        """
        if self.aspect_pipeline is None:
            warnings.warn("[TransformerABSA] Aspect extraction model not loaded; returning empty list.")
            return []

        preds = self.aspect_pipeline(text)
        aspects = []
        for p in preds:
            if "ASP" not in p.get("entity_group", ""):
                continue
            aspect = p["word"].strip()
            start, end = p["start"], p["end"]
            conf = float(p["score"])
            aspects.append((aspect, (start, end), conf))
        return aspects

    # --------------------------------------------------------
    # Sentiment Prediction per Aspect
    # --------------------------------------------------------
    def _predict_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:
        """
        Predict sentiment polarity for a specific aspect within a review.

        Args:
            text (str): The full review text.
            aspect (str): The aspect term to classify sentiment for.

        Returns:
            Tuple of (sentiment_label, confidence)
            Example: ("positive", 0.87)
        """
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            warnings.warn("[TransformerABSA] Sentiment model not loaded; returning neutral by default.")
            return "neutral", 0.0

        inputs = self.sentiment_tokenizer(
            text,
            aspect,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        if self.cfg.device >= 0:
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        label_id = int(probs.argmax())
        sentiment = self.label_map.get(label_id, "neutral")
        conf = float(probs[label_id])
        return sentiment, conf

    # --------------------------------------------------------
    # Main ABSA Pipeline
    # --------------------------------------------------------
    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Perform full ABSA analysis on input text.

        Pipeline:
            1. Extract aspect terms using the token-classification model.
            2. For each aspect, predict sentiment polarity using the classifier.
            3. Aggregate results and remove duplicates.

        Args:
            text (str): Input review or sentence.

        Returns:
            List[AspectSentiment]: Aspect-sentiment predictions with confidence.
        """
        aspect_preds = self._extract_aspects(text)
        results: List[AspectSentiment] = []

        for aspect_text, span, asp_conf in aspect_preds:
            sentiment, conf = self._predict_sentiment(text, aspect_text)

            # Hybrid confidence rule
            if conf < self.cfg.sentiment_conf_threshold and asp_conf > conf:
                conf = asp_conf

            results.append(
                AspectSentiment(
                    aspect=aspect_text,
                    sentiment=sentiment,
                    confidence=round(conf, 2),
                    text_span=span,
                )
            )

        # Deduplicate overlapping aspects (keep highest confidence)
        dedup = {}
        for r in results:
            if r.aspect not in dedup or r.confidence > dedup[r.aspect].confidence:
                dedup[r.aspect] = r

        return list(dedup.values())


# --------------------------------------------------------
# Example Run
# --------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    analyzer = TransformerABSA()

    df = pd.read_json("../data/yelp_sample_20k.json", lines=True)
    sample = [
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

    for i, review in enumerate(sample, 1):
        print(f"\nReview {i}: {review}\n")
        aspects = analyzer.analyze(review)

        if not aspects:
            print("  No clear aspects detected.")
        else:
            for asp in aspects:
                print(f"   ➜ {asp.aspect:<25} | {asp.sentiment:<10} | conf={asp.confidence:.2f}")
