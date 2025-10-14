from src.lexicon_absa import LexiconABSA
import os
from src.transformer_absa import TransformerABSA
import pandas as pd
from src.llm_absa import LLMABSA
import pytest

# Dynamically resolve the data file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # The project root
DATA_PATH = os.path.join(BASE_DIR, "data", "yelp_sample.json")

# Load your dataset
df = pd.read_json(DATA_PATH, lines=True)
sample_reviews = df["text"]


@pytest.mark.parametrize("analyzer_class", [LexiconABSA, TransformerABSA, LLMABSA])
def test_model(analyzer_class):

    analyzer = analyzer_class()
    for i, review in enumerate(sample_reviews, 1):
        print(f"\n📝 Review {i}: {review[:180]}...")
        aspects = analyzer.analyze(review)
        if not aspects:
            print("   ⚠️ No clear aspects detected.")
        else:
            for asp in aspects:
                print(f"   ➜ {asp.aspect:<15} | {asp.sentiment:<8} | conf={asp.confidence:.2f}")

if __name__ == "__main__":
    print("=== Lexicon-Based ABSA ===")
    test_model(LexiconABSA)

    print("\n=== Transformer ABSA ===")
    test_model(TransformerABSA)

    print("\n=== Local LLM (Ollama) ABSA ===")
    test_model(LLMABSA)

