from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from typing import List
from pyabsa import ATEPCCheckpointManager
from src.base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    """Implementation 2 — Pre-trained Transformer ABSA model."""

    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
        print("Loading pre-trained model. This may take a moment…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # <--- KEY CHANGE
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and return a list of aspect-sentiment predictions."""
        doc = self.spacy_nlp(text)
        aspects = [t.text for t in doc if t.pos_ == "NOUN"]

        results = []
        for asp in aspects:
            inputs = self.tokenizer(text, asp, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                label_id = torch.argmax(probs).item()

            sentiment = self.label_map[label_id]
            confidence = float(probs[label_id])
            span = (text.find(asp), text.find(asp) + len(asp))

            results.append(AspectSentiment(aspect=asp, sentiment=sentiment, confidence=confidence, text_span=span))
        return results


if __name__ == "__main__":
    model = TransformerABSA()
    sample = "The camera quality is amazing but the battery life is disappointing."
    for r in model.analyze(sample):
        print(f"{r.aspect:<12} → {r.sentiment:<9} ({r.confidence:.2f})")