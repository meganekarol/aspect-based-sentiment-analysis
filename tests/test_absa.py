import os
import pandas as pd
import pytest
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import TransformerABSA
from src.llm_absa import LLMABSA

# Dynamically resolve the data file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # The project root
DATA_PATH = os.path.join(BASE_DIR, "data", "yelp_sample_20k.json")

# Load your dataset
df = pd.read_json(DATA_PATH, lines=True)
sample_reviews = df["text"].sample(5, random_state=42)


@pytest.mark.parametrize("analyzer_class", [LexiconABSA, TransformerABSA, LLMABSA])
def test_model(analyzer_class):
    """Parameterized pytest function to evaluate multiple ABSA analyzers."""
    analyzer = analyzer_class()
    for i, review in enumerate(sample_reviews, 1):
        print(f"\nReview {i}: {review[:180]}...")
        aspects = analyzer.analyze(review)
        if not aspects:
            print("  No clear aspects detected.")
        else:
            for asp in aspects:
                print(f"   ➜ {asp.aspect:<15} | {asp.sentiment:<8} | conf={asp.confidence:.2f}")


def run_manual_test():
    """Manual run for LexiconABSA without pytest."""
    analyzer = LexiconABSA()
    print("=== Lexicon-Based ABSA ===")
    for i, review in enumerate(sample_reviews, 1):
        print(f"\nReview {i}: {review[:180]}...")
        aspects = analyzer.analyze(review)
        if not aspects:
            print("  No clear aspects detected.")
        else:
            for asp in aspects:
                print(f"   ➜ {asp.aspect:<15} | {asp.sentiment:<8} | conf={asp.confidence:.2f}")


if __name__ == "__main__":
    run_manual_test()
