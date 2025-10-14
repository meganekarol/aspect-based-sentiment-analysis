"""
Implementation 3 — LLM-Based ABSA using Ollama
"""

from ollama import chat
from src.base import ABSAAnalyzer, AspectSentiment
from typing import List
import json
import time


class LLMABSA(ABSAAnalyzer):
    """
    Aspect-Based Sentiment Analysis using a local LLM (Ollama).
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        print(f"✅ LLMABSA initialized using model: {self.model_name}")

        # A simple few-shot prompt to teach the model the task
        self.base_prompt = """
You are an expert in aspect-based sentiment analysis (ABSA).
Given a review, extract all aspects (topics or entities) and classify the sentiment
toward each aspect as Positive, Negative, or Neutral.
Return your answer strictly in JSON format as a list of objects like:
[
  {"aspect": "pizza", "sentiment": "positive"},
  {"aspect": "service", "sentiment": "negative"}
]

Example:
Input: "The pizza was delicious but the service was terrible."
Output:
[
  {"aspect": "pizza", "sentiment": "positive"},
  {"aspect": "service", "sentiment": "negative"}
]
"""

    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text using local LLM and return aspect-sentiment pairs."""
        prompt = self.base_prompt + f'\nNow analyze this text:\n"{text}"\n'

        try:
            response = chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            raw_output = response["message"]["content"].strip()

            # Try to parse JSON safely
            try:
                data = json.loads(raw_output)
            except json.JSONDecodeError:
                # If the LLM returns text with extra commentary, try to clean it
                json_start = raw_output.find("[")
                json_end = raw_output.rfind("]") + 1
                cleaned = raw_output[json_start:json_end]
                data = json.loads(cleaned)

            results = []
            for item in data:
                aspect = item.get("aspect", "").strip()
                sentiment = item.get("sentiment", "").lower()
                confidence = 1.0  # Ollama doesn’t provide probabilities
                span = (text.find(aspect), text.find(aspect) + len(aspect))
                results.append(AspectSentiment(aspect, sentiment, confidence, span))

            return results

        except Exception as e:
            print(f"❌ Error during LLM analysis: {e}")
            return []


if __name__ == "__main__":
    model = LLMABSA(model_name="llama3")  # You can replace with "mistral", "phi3", etc.
    text = "The camera quality is amazing but the battery drains too fast."
    results = model.analyze(text)

    print("\n💬 Input:", text)
    print("🔍 Output:")
    for r in results:
        print(f"   ➜ Aspect: {r.aspect:15} | Sentiment: {r.sentiment:9} | Confidence: {r.confidence:.2f}")
