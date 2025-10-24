"""
TransformerABSA — Hybrid Transformer Implementation
---------------------------------------------------
Uses:
- yangheng/deberta-v3-base-end2end-absa → for aspect extraction
- yangheng/deberta-v3-base-absa-v1.1 → for sentiment classification

This implementation jointly leverages both models:
    • High aspect recall (from End2End model)
    • High sentiment accuracy (from ABSA model)
"""

from dataclasses import dataclass
from typing import List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from base import ABSAAnalyzer, AspectSentiment


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class TransformerABSAConfig:
    aspect_model: str = "yangheng/deberta-v3-base-end2end-absa"
    sentiment_model: str = "yangheng/deberta-v3-base-absa-v1.1"
    sentiment_conf_threshold: float = 0.6
    aggregation_strategy: str = "simple"
    device: int = 0 if torch.cuda.is_available() else -1


# ------------------------------------------------------------
# Implementation
# ------------------------------------------------------------
class TransformerABSA(ABSAAnalyzer):
    def __init__(self, cfg: TransformerABSAConfig | None = None):
        self.cfg = cfg or TransformerABSAConfig()

        # Aspect extraction model (End-to-End ABSA)
        self.aspect_pipeline = pipeline(
            "token-classification",
            model=self.cfg.aspect_model,
            aggregation_strategy=self.cfg.aggregation_strategy,
            device=self.cfg.device,
        )

        # Sentiment classification model (Aspect-conditioned)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.cfg.sentiment_model)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.sentiment_model)

        # Sentiment label mapping
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    # --------------------------------------------------------
    # Aspect Extraction
    # --------------------------------------------------------
    def _extract_aspects(self, text: str) -> List[Tuple[str, Tuple[int, int], float]]:
        preds = self.aspect_pipeline(text)
        aspects = []
        for p in preds:
            if "ASP" not in p["entity_group"]:
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
        Feeds the text and aspect into the sentiment classification model.
        """
        # Format input as expected by ABSA model: [CLS] text [SEP] aspect [SEP]
        inputs = self.sentiment_tokenizer(
            text,
            aspect,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )

        if self.cfg.device >= 0:
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            self.sentiment_model.to(self.cfg.device)

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        label_id = int(probs.argmax())
        sentiment = self.label_map[label_id]
        conf = float(probs[label_id])
        return sentiment, conf

    # --------------------------------------------------------
    # Main ABSA pipeline
    # --------------------------------------------------------
    def analyze(self, text: str) -> List[AspectSentiment]:
        aspect_preds = self._extract_aspects(text)
        results: List[AspectSentiment] = []

        for aspect_text, span, asp_conf in aspect_preds:
            sentiment, conf = self._predict_sentiment(text, aspect_text)

            # Hybrid merge rule
            if conf < self.cfg.sentiment_conf_threshold and asp_conf > conf:
                conf = asp_conf  # fallback to aspect model confidence

            results.append(
                AspectSentiment(
                    aspect=aspect_text,
                    sentiment=sentiment,
                    confidence=round(conf, 2),
                    text_span=span,
                )
            )

        # Deduplicate overlapping aspects
        dedup = {}
        for r in results:
            if r.aspect not in dedup or r.confidence > dedup[r.aspect].confidence:
                dedup[r.aspect] = r

        return list(dedup.values())

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

    # analyzer = TransformerABSA()
    # text = """Oh perfect, another burnt pizza. Amazing job."""
    # results = analyzer.analyze(text)
    # for r in results:
    #     print(f"{r.aspect:<20} | {r.sentiment:<8} | conf={r.confidence:.2f}")

