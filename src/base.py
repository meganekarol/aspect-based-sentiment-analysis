from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str  # 'positive'|'negative'|'neutral'
    confidence: float  # 0.0 - 1.0
    text_span: Tuple[int,int] = None

class ABSAAnalyzer:
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Return list of AspectSentiment"""
        raise NotImplementedError
