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

import warnings
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from absa.base import ABSAAnalyzer, AspectSentiment


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

    def __init__(self, cfg: TransformerABSAConfig | None = None):
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
                stacklevel=2,
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
                stacklevel=2,
            )
            return None, None

    # --------------------------------------------------------
    # Aspect Extraction
    # --------------------------------------------------------
    def _extract_aspects(self, text: str) -> list[tuple[str, tuple[int, int], float]]:
        """
        Extract aspect terms from text using a token-classification model.

        Args:
            text (str): Input review text.

        Returns:
            List of tuples (aspect_text, (start_char, end_char), confidence)
            Example: [("service", (12, 19), 0.95)]
        """
        if self.aspect_pipeline is None:
            warnings.warn(
                "[TransformerABSA] Aspect extraction model not loaded; returning empty list.", stacklevel=2
            )
            return []

        preds = self.aspect_pipeline(text)
        aspects: list[tuple[str, tuple[int, int], float]] = []
        for p in sorted(preds, key=lambda p: p["start"]):
            if "ASP" not in p.get("entity_group", ""):
                continue
            start, end = p["start"], p["end"]
            # SentencePiece offsets include the leading space ("▁word"); trim it.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start == end:
                continue
            # Sub-word groups can stop mid-word ("staff memeber|s"); extend to word boundaries.
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            conf = float(p["score"])

            # The "simple" aggregation splits one aspect into adjacent groups when
            # its tokens carry different polarity tags (B-ASP-Positive, I-ASP-Neutral):
            # "outdoor" + "seating". Merge groups separated only by whitespace.
            if aspects and not text[aspects[-1][1][1] : start].strip():
                _, (prev_start, _), prev_conf = aspects.pop()
                start, conf = prev_start, min(prev_conf, conf)
            aspects.append((text[start:end], (start, end), conf))
        return aspects

    # --------------------------------------------------------
    # Sentiment Prediction per Aspect
    # --------------------------------------------------------
    def _predict_sentiment(self, text: str, aspect: str) -> tuple[str, float]:
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
            warnings.warn(
                "[TransformerABSA] Sentiment model not loaded; returning neutral by default.", stacklevel=2
            )
            return "neutral", 0.0

        inputs = self.sentiment_tokenizer(
            text, aspect, max_length=512, truncation=True, padding=True, return_tensors="pt"
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
    def analyze(self, text: str) -> list[AspectSentiment]:
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
        results: list[AspectSentiment] = []

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
