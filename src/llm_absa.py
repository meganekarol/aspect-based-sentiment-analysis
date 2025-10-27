"""
Implementation 3 — LLM-Based ABSA using Ollama
------------------------------------------------
Uses a local LLM (e.g., Llama3, Mistral) to perform
Aspect-Based Sentiment Analysis via structured prompting.

Features:
- Few-shot prompt for stable ABSA behavior
- Confidence and text span estimation
- Unified API: returns List[AspectSentiment]
"""
import json
import time
from typing import List, Optional, Tuple
from ollama import chat
from src.base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):
    """
    Aspect-Based Sentiment Analysis using a locally running LLM via Ollama.
    """

    def __init__(self, model_name: str = "llama3", max_aspects: int = 8):
        self.model_name = model_name
        self.max_aspects = max_aspects
        print(f"LLMABSA initialized with model: {self.model_name}")

        # --- Base System Prompt ---
        self.system_prompt = f"""
You are an expert in Aspect-Based Sentiment Analysis (ABSA).

Given a customer review, extract *aspects* (nouns or noun phrases describing concrete entities, e.g.,
"service", "chicken tikka masala", "outdoor seating", "baristas", "parking lot")
and classify sentiment toward each as one of:
- positive
- negative
- neutral

Return your answer *strictly in JSON* as a list of objects:

[
  {{
    "aspect": "string",
    "sentiment": "positive | negative | neutral",
    "confidence": float  # between 0.0 and 1.0
  }}
]

Rules:
- Join multiword noun phrases (e.g. "chicken tikka masala", "customer service").
- Ignore vague or generic aspects like "thing", "experience", "place", "time".
- If uncertain, choose 'neutral' with confidence ~0.5.
- Limit to at most {self.max_aspects} aspects.
"""

        # --- Few-shot examples ---
        self.few_shots = [
            {
                "role": "user",
                "content": "The mushroom risotto was perfection, but the service was slow. The atmosphere was cozy and the chocolate cake divine."
            },
            {
                "role": "assistant",
                "content": json.dumps([
                    {"aspect": "mushroom risotto", "sentiment": "positive", "confidence": 0.9},
                    {"aspect": "service", "sentiment": "negative", "confidence": 0.85},
                    {"aspect": "atmosphere", "sentiment": "positive", "confidence": 0.8},
                    {"aspect": "chocolate cake", "sentiment": "positive", "confidence": 0.9},
                ])
            },
            {
                "role": "user",
                "content": "I tried their decaf latte which was tasty and foamy. The baristas were friendly. The parking lot is always packed."
            },
            {
                "role": "assistant",
                "content": json.dumps([
                    {"aspect": "decaf latte", "sentiment": "positive", "confidence": 0.8},
                    {"aspect": "baristas", "sentiment": "positive", "confidence": 0.85},
                    {"aspect": "parking lot", "sentiment": "negative", "confidence": 0.75},
                ])
            },
        ]

    # -------------------------------------------------------------------------
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Perform ABSA using the local Ollama model with retry logic."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.few_shots,
            {"role": "user", "content": f"Text:\n{text}\n\nReturn ONLY the JSON array as instructed."},
        ]

        max_retries = 3
        delay = 1  # start with 1 second delay

        for attempt in range(1, max_retries + 1):
            try:
                response = chat(model=self.model_name, messages=messages)
                raw_output = response["message"]["content"].strip()

                # --- Parse JSON robustly ---
                data = self._safe_json_parse(raw_output)
                results = self._normalize_output(data, text)
                return results

            except Exception as e:
                print(f"[LLMABSA] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                else:
                    print("[LLMABSA] All retries failed.")
                    return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_json_parse(raw_output: str) -> list:
        """Attempt to clean and parse LLM JSON output robustly."""
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            try:
                start, end = raw_output.find("["), raw_output.rfind("]") + 1
                if start >= 0 and end > 0:
                    return json.loads(raw_output[start:end])
            except Exception:
                pass
        return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_output(data: list, text: str) -> List[AspectSentiment]:
        """Normalize parsed JSON into AspectSentiment objects."""
        results = []
        for item in data:
            aspect = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "neutral")).lower()
            if not aspect:
                continue
            conf = float(item.get("confidence", 0.7))
            span = LLMABSA._find_span(text, aspect)
            results.append(AspectSentiment(aspect, sentiment, conf, span))
        return results

    # -------------------------------------------------------------------------
    @staticmethod
    def _find_span(text: str, aspect: str) -> Optional[Tuple[int, int]]:
        """Find the first occurrence of an aspect in the text."""
        start = text.lower().find(aspect.lower())
        if start == -1:
            return None
        return (start, start + len(aspect))

if __name__ == "__main__":
    model = LLMABSA(model_name="llama3", max_aspects=8)
    text = "The coffee was bitter and cold. However, the baristas were very friendly."
    results = model.analyze(text)

    for r in results:
        print(f"   ➜ {r.aspect:<20} | {r.sentiment:<10} | conf={r.confidence:.2f}")