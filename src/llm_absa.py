"""
Implementation 3 — LLM-Based ABSA using Ollama
------------------------------------------------
Uses a local LLM (e.g., Llama3, Mistral) to perform
Aspect-Based Sentiment Analysis via structured prompting.

Features:
- Few-shot prompt for stable ABSA behavior
- Strict JSON schema for consistent output
- Automatic fallback parsing for malformed JSON
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

    def __init__(
        self,
        model_name: str = "llama3",
        max_aspects: int = 8,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.max_aspects = max_aspects
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
- Return ONLY valid JSON, no additional text.
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
            {
                "role": "user",
                "content": "The pizza was burnt and tasted terrible. The waiter was rude and didn't apologize."
            },
            {
                "role": "assistant",
                "content": json.dumps([
                    {"aspect": "pizza", "sentiment": "negative", "confidence": 0.95},
                    {"aspect": "waiter", "sentiment": "negative", "confidence": 0.9},
                ])
            },
        ]

    # -------------------------------------------------------------------------
    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Perform ABSA using the local Ollama model with retry logic.

        Args:
            text: Input text to analyze

        Returns:
            List of AspectSentiment objects
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                results = self._attempt_analysis(text)

                # Only return if we got valid results
                if results:
                    if attempt > 0:
                        print(f"[LLMABSA] Succeeded on attempt {attempt + 1}")
                    return results
                else:
                    print(f"[LLMABSA] Attempt {attempt + 1}: No valid results parsed")

            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                print(f"[LLMABSA] Attempt {attempt + 1}: {last_error}")

            except ConnectionError as e:
                last_error = f"Connection error: {e}"
                print(f"[LLMABSA] Attempt {attempt + 1}: {last_error}")

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                print(f"[LLMABSA] Attempt {attempt + 1}: {last_error}")

            # Wait before retrying (except on last attempt)
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        # All retries exhausted
        print(f"[LLMABSA] ✗ All {self.max_retries} attempts failed. Last error: {last_error}")
        return []

    # -------------------------------------------------------------------------
    def _attempt_analysis(self, text: str) -> List[AspectSentiment]:
        """Single attempt at analysis."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.few_shots,
            {"role": "user", "content": f"Text:\n{text}\n\nReturn ONLY the JSON array as instructed."},
        ]

        # Use format='json' for newer Ollama versions (forces JSON output)
        response = chat(
            model=self.model_name,
            messages=messages,
            format='json',  # Ensures JSON-formatted output
            options={
                'temperature': 0.3,  # Lower temperature for more consistent output
            }
        )

        raw_output = response["message"]["content"].strip()

        # --- Parse JSON robustly ---
        data = self._safe_json_parse(raw_output)

        if not data:
            raise json.JSONDecodeError("Empty or invalid JSON", raw_output, 0)

        results = self._normalize_output(data, text)
        return results

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_json_parse(raw_output: str) -> list:
        """Attempt to clean and parse LLM JSON output robustly."""
        # First, try direct parsing
        try:
            parsed = json.loads(raw_output)
            # Ensure it's a list
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and 'aspects' in parsed:
                # Handle case where LLM wraps in object
                return parsed['aspects']
        except json.JSONDecodeError:
            pass

        # Second, try to extract JSON array from text
        try:
            start = raw_output.find("[")
            end = raw_output.rfind("]") + 1
            if start >= 0 and end > 0:
                extracted = raw_output[start:end]
                parsed = json.loads(extracted)
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            pass

        return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_output(data: list, text: str) -> List[AspectSentiment]:
        """Normalize parsed JSON into AspectSentiment objects."""
        results = []
        for item in data:
            if not isinstance(item, dict):
                continue

            aspect = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "neutral")).lower()

            if not aspect:
                continue

            # Validate sentiment value
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'

            conf = float(item.get("confidence", 0.7))
            # Clamp confidence to valid range
            conf = max(0.0, min(1.0, conf))

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