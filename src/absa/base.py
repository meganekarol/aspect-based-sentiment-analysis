"""Shared data model and interface for every ABSA implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

SENTIMENTS = ("positive", "negative", "neutral")


@dataclass
class AspectSentiment:
    """One aspect found in a text and the sentiment expressed towards it."""

    aspect: str  # The aspect/feature mentioned, e.g. "service"
    sentiment: str  # 'positive' | 'negative' | 'neutral'
    confidence: float  # Confidence score in [0.0, 1.0]
    text_span: tuple[int, int] | None = None  # (start, end) char offsets in the original text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ABSAAnalyzer(ABC):
    """Base interface for all ABSA implementations."""

    @abstractmethod
    def analyze(self, text: str) -> list[AspectSentiment]:
        """Analyze ``text`` and return its aspect-sentiment pairs."""
