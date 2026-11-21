"""
LLM-based ABSA using a local model served by Ollama.

Uses a local LLM (e.g. Llama 3, Mistral) with a few-shot prompt that asks
for a strict JSON answer, then normalizes that answer into
``AspectSentiment`` objects.

- Few-shot prompt for stable behaviour
- Defensive JSON parsing + label/confidence validation
- Retries with exponential backoff
- Ollama host configurable via ``host=`` or the ``OLLAMA_HOST`` env var
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ollama import Client, chat

from absa.base import SENTIMENTS, ABSAAnalyzer, AspectSentiment

logger = logging.getLogger(__name__)


class LLMABSA(ABSAAnalyzer):
    """Aspect-Based Sentiment Analysis using a locally running LLM via Ollama."""

    def __init__(
        self,
        model_name: str = "llama3",
        max_aspects: int = 8,
        host: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.max_aspects = max_aspects
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Without an explicit host, ollama's module-level client honours OLLAMA_HOST.
        self._client_chat = Client(host=host).chat if host else None
        logger.info("LLMABSA initialized with model: %s", self.model_name)

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

        self.few_shots = [
            {
                "role": "user",
                "content": "The mushroom risotto was perfection, but the service was slow. "
                "The atmosphere was cozy and the chocolate cake divine.",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {"aspect": "mushroom risotto", "sentiment": "positive", "confidence": 0.9},
                        {"aspect": "service", "sentiment": "negative", "confidence": 0.85},
                        {"aspect": "atmosphere", "sentiment": "positive", "confidence": 0.8},
                        {"aspect": "chocolate cake", "sentiment": "positive", "confidence": 0.9},
                    ]
                ),
            },
            {
                "role": "user",
                "content": "I tried their decaf latte which was tasty and foamy. The baristas were "
                "friendly. The parking lot is always packed.",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {"aspect": "decaf latte", "sentiment": "positive", "confidence": 0.8},
                        {"aspect": "baristas", "sentiment": "positive", "confidence": 0.85},
                        {"aspect": "parking lot", "sentiment": "negative", "confidence": 0.75},
                    ]
                ),
            },
        ]

    def _chat(self, **kwargs: Any) -> Any:
        return (self._client_chat or chat)(**kwargs)

    def analyze(self, text: str) -> list[AspectSentiment]:
        """Perform ABSA using the local Ollama model, retrying on failure."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.few_shots,
            {"role": "user", "content": f"Text:\n{text}\n\nReturn ONLY the JSON array as instructed."},
        ]

        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._chat(model=self.model_name, messages=messages)
                raw_output = response["message"]["content"].strip()
                data = self._safe_json_parse(raw_output)
                return self._normalize_output(data, text)[: self.max_aspects]
            except Exception as e:  # noqa: BLE001 - any client/parse error triggers a retry
                logger.warning("[LLMABSA] Attempt %d/%d failed: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
        logger.error("[LLMABSA] All retries failed.")
        return []

    @staticmethod
    def _safe_json_parse(raw_output: str) -> list:
        """Parse the model's JSON answer, tolerating text around the array.

        Raises ``ValueError`` when no JSON array can be recovered, so that
        ``analyze`` retries instead of silently returning nothing.
        """
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            start, end = raw_output.find("["), raw_output.rfind("]") + 1
            if start < 0 or end <= start:
                raise ValueError("no JSON array in model output") from None
            data = json.loads(raw_output[start:end])
        if isinstance(data, dict):  # some models wrap the list: {"aspects": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            raise ValueError(f"expected a JSON list, got {type(data).__name__}")
        return data

    @staticmethod
    def _normalize_output(data: list, text: str) -> list[AspectSentiment]:
        """Turn parsed JSON items into validated ``AspectSentiment`` objects."""
        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect", "")).strip()
            if not aspect:
                continue
            sentiment = str(item.get("sentiment", "neutral")).strip().lower()
            if sentiment not in SENTIMENTS:
                sentiment = "neutral"
            try:
                conf = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            conf = min(max(conf, 0.0), 1.0)
            span = LLMABSA._find_span(text, aspect)
            results.append(AspectSentiment(aspect, sentiment, conf, span))
        return results

    @staticmethod
    def _find_span(text: str, aspect: str) -> tuple[int, int] | None:
        """Find the first occurrence of an aspect in the text."""
        start = text.lower().find(aspect.lower())
        if start == -1:
            return None
        return (start, start + len(aspect))
