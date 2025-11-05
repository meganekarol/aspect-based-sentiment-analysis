import unittest
from src.lexicon_absa import LexiconABSA,LexiconABSAConfig
from src.transformer_absa import TransformerABSA ,TransformerABSAConfig
from src.llm_absa import LLMABSA


class TestAllABSA(unittest.TestCase):
    def setUp(self):
        # Long review to test
        self.long_review = """
Food was good tasting, portions felt small -especially side dishes. I guess they are trying to do gourmet comfort food, 
but something didn't mesh. It was sorta family style served, 
but we felt bad taking more than a tablespoon of any dish because there 
wouldn't be enough to share. Not sure what audience they are going for
        """

        # Initialize all analyzers
        self.lexicon_analyzer = LexiconABSA(LexiconABSAConfig())
        self.transformer_analyzer = TransformerABSA(TransformerABSAConfig())
        self.llm_analyzer = LLMABSA(model_name="llama3")

    def test_all_models(self):
        analyzers = {
            "Lexicon-based": self.lexicon_analyzer,
            "Transformer-based": self.transformer_analyzer,
            "LLM (Ollama)": self.llm_analyzer
        }

        for name, analyzer in analyzers.items():
            print(f"\n--- Running {name} ABSA ---")
            results = analyzer.analyze(self.long_review)
            if not results:
                print("No aspects detected.")
            else:
                for r in results:
                    print(
                        f"Aspect: {r.aspect}, Sentiment: {r.sentiment}, Confidence: {getattr(r, 'confidence', 'N/A')}")


if __name__ == "__main__":
    unittest.main()