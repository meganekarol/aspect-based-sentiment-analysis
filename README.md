# LLM Sentiment Project 2025  (Group 17)
**Aspect-Based Sentiment Analysis (ABSA) System**  
*Course: Applied Computer Science Data & AI 6*  
*Authors: Lesly Tianga, Megane Fomo*
---

## Project Overview

This project implements a comprehensive **Aspect-Based Sentiment Analysis (ABSA)** system that identifies aspects (features) mentioned in text and determines the sentiment expressed toward each aspect.

Unlike traditional sentiment analysis, which assigns a single polarity to an entire review, **ABSA** breaks the text into *aspect–sentiment pairs* for more fine-grained insights.

**Example:**

> "The pizza was delicious but the service was terrible."

| Aspect | Sentiment | Confidence |
|---------|------------|-------------|
| pizza   | positive   | 0.92        |
| service | negative   | 0.87        |

The system includes **three independent implementations** of ABSA, each using a different approach and unified through a shared API:

| Implementation | Description | Core Technology |
|----------------|--------------|-----------------|
| **LexiconABSA** | Rule-based ABSA using linguistic rules and sentiment lexicons | spaCy + VADER |
| **TransformerABSA** | Pre-trained ABSA model using Transformers | DeBERTa-v3 (Hugging Face) |
| **LLMABSA** | Aspect–sentiment extraction using local large language models | Ollama (Llama/Mistral) |

---

## Project Structure
project/
├── README.md
├── requirements.txt
├── data/
│ ├── yelp_sample_20k.json
│ ├── yelp_academic_dataset_review.json
│ 
├── src/
│ ├── base.py
│ ├── lexicon_absa.py
│ ├── transformer_absa.py
│ ├── llm_absa.py
│ ├── utils.py
├── notebooks/
│ ├── exploration.ipynb
│ └── comparison.ipynb
└── tests/
├── test_lexicon_absa.py
└── test_transformer_absa.py 
└── test_llama_absa.py

---

## Setup Instructions

### 1. Download and Unzip Project

```
Download the zip file and unzip in your favourite location
cd llm-sentiment-project
```
### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download NLP models
```
python -m spacy download en_core_web_sm
```
Optional(for TransformerABSA):
```
pip install sentencepiece protobuf
```

## Dependencies
| Library                           | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| **spaCy**                         | Linguistic preprocessing, dependency parsing |
| **vaderSentiment**                | Lexicon-based sentiment scoring              |
| **transformers**                  | Pre-trained ABSA model (DeBERTa-v3)          |
| **torch**                         | Model inference for Transformers             |
| **ollama**                        | Local LLM-based ABSA                         |
| **pandas / seaborn / matplotlib** | Exploratory data analysis and visualization  |
| **pytest**                        | Unit testing framework                       |

## Usage Examples
### 1. LexiconABSA (Rule-Based Approach)
```bash
model = LexiconABSA(LexiconABSAConfig())
text = "The food was great but the service was slow."
results = model.analyze(text)
for r in results:
    print(f"   ➜ {r.aspect:<20} | {r.sentiment:<10} | conf={r.confidence:.2f}")
```
Output:
```
➜ food      | positive   | conf=0.46
➜ service   | negative   | conf=0.38
```
### 2. TransformerABSA (Pre-trained Model Approach)
```bash
model = TransformerABSA(TransformerABSAConfig())
text = "The mushroom risotto was perfection, but the service was slow. The atmosphere was cozy"
results = model.analyze(text)
for r in results:
    print(f"   ➜ {r.aspect:<20} | {r.sentiment:<10} | conf={r.confidence:.2f}")
```
Output:
```
pizza positive 0.94
price negative 0.83
```

### 3. LLMABSA (Large Language Model Approach)
Make sure Ollama is running
``` 
ollama run llama3
```
```python
from src.llm_absa import LLMABSA
model = LLMABSA("llama3")
text = "I tried their decaf latte which was tasty and foamy. The baristas were friendly but the parking lot was small."
for r in model.analyze(text):
    print(r)
```
Output:
```
pizza positive 0.94
price negative 0.83
```
## Running Tests
Each implementation has its own test file under /tests:
- test_lexicon_absa.py 
- test_transformer_absa.py 
- test_llama_absa.py

Run all tests
``` 
pytest -v
```
Run one test file
``` 
pytest -v tests\test_lexicon_absa.py 
```


## Unified API Documentation
All three implementations inherit from the same base interface defined in src/base.py.

### Base Class

```python
class AspectSentiment:
    aspect: str # The aspect/feature mentioned
    sentiment: str  # 'positive'|'negative'|'neutral'
    confidence: float  # Confidence score (0.0 to 1.0)
    text_span: Tuple[int,int] = None # Optional: (start, end) position in original text

class ABSAAnalyzer:
    """Base class/interface for all ABSA implementations"""
    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs
        Args:
        text: Input text to analyze
        Returns:
        List of AspectSentiment objects
        """
    raise NotImplementedError
```

## Design Decisions & Rationale
- 


## References
*spaCy Documentation:* https://spacy.io/  
*VADER Sentiment:* https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/  
*Hugging Face Transformers:* https://huggingface.co/learn/llm-course/chapter0/1, https://huggingface.co/yangheng/deberta-v3-large-absa-v1.1, https://huggingface.co/yangheng/deberta-v3-base-end2end-absa  
*Ollama Local LLMs:* https://github.com/ollama/ollama/blob/main/docs/api.md  
*Others:* https://www.geeksforgeeks.org/machine-learning/what-is-sentiment-analysis/, https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf,https://universaldependencies.org/u/dep/  
*Testing:* https://docs.pytest.org/en/stable/  
*Code:* ChatGPT
