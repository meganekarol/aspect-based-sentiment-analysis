"""Aspect-Based Sentiment Analysis with three interchangeable back-ends.

The analyzers are imported lazily, so ``import absa`` only needs the
dependencies of the back-end you actually use (e.g. no torch for the
lexicon model).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from absa.base import SENTIMENTS, ABSAAnalyzer, AspectSentiment

__version__ = "1.0.0"

_LAZY = {
    "LexiconABSA": "absa.lexicon_absa",
    "LexiconABSAConfig": "absa.lexicon_absa",
    "TransformerABSA": "absa.transformer_absa",
    "TransformerABSAConfig": "absa.transformer_absa",
    "LLMABSA": "absa.llm_absa",
}

__all__ = [
    "SENTIMENTS",
    "ABSAAnalyzer",
    "AspectSentiment",
    "get_analyzer",
    "LexiconABSA",
    "LexiconABSAConfig",
    "TransformerABSA",
    "TransformerABSAConfig",
    "LLMABSA",
]

if TYPE_CHECKING:
    from absa.lexicon_absa import LexiconABSA, LexiconABSAConfig
    from absa.llm_absa import LLMABSA
    from absa.transformer_absa import TransformerABSA, TransformerABSAConfig


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'absa' has no attribute {name!r}")


def get_analyzer(name: str, **kwargs: Any) -> ABSAAnalyzer:
    """Build an analyzer by short name: ``lexicon``, ``transformer`` or ``llm``."""
    name = name.lower()
    if name == "lexicon":
        from absa.lexicon_absa import LexiconABSA

        return LexiconABSA(**kwargs)
    if name == "transformer":
        from absa.transformer_absa import TransformerABSA

        return TransformerABSA(**kwargs)
    if name == "llm":
        from absa.llm_absa import LLMABSA

        return LLMABSA(**kwargs)
    raise ValueError(f"Unknown analyzer {name!r}; choose lexicon, transformer or llm")
