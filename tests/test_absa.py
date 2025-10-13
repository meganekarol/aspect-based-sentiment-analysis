from src.lexicon_absa import LexiconABSA
import random
import json

sample_reviews = []
with open("../data/yelp_sample_20k.json", "r", encoding="utf-8") as f:
    all_lines = f.readlines()
    sample_lines = random.sample(all_lines, 5)  # pick 5 random reviews
    for line in sample_lines:
        data = json.loads(line)
        sample_reviews.append(data["text"])

def test_basic_aspect_sentiment():
    # --- Initialize analyzer ---
    analyzer = LexiconABSA()

    results_table = []

    for i, review in enumerate(sample_reviews, 1):
        print(f"\n📝 Review {i}: {review[:180]}...")
        aspects = analyzer.analyze(review)

        if not aspects:
            print("   ⚠️ No clear aspects detected.")
        else:
            for asp in aspects:
                results_table.append({
                    "Review #": i,
                    "Aspect": asp.aspect,
                    "Sentiment": asp.sentiment,
                    "Confidence": asp.confidence
                })
                print(f"   ➜ {asp.aspect:<15} | {asp.sentiment:<8} | conf={asp.confidence}")

if __name__ == "__main__":
    test_basic_aspect_sentiment()