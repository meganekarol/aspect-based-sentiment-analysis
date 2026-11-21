"""Loaders for labelled ABSA benchmarks.

SemEval-2014 Task 4 (Pontiki et al., 2014) is the standard aspect-term
benchmark. The data is not redistributed in this repo; ``download_semeval``
fetches the official gold XML from a Hugging Face mirror into ``data/semeval/``.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SEMEVAL_URLS = {
    "restaurants": "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/main/"
    "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
    "laptops": "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/main/"
    "ABSA_Gold_TestData/Laptops_Test_Gold.xml",
}
DEFAULT_DIR = Path("data/semeval")


@dataclass
class GoldAspect:
    term: str
    polarity: str  # positive | negative | neutral | conflict
    span: tuple[int, int]


@dataclass
class GoldSentence:
    id: str
    text: str
    aspects: list[GoldAspect] = field(default_factory=list)


def download_semeval(domain: str = "restaurants", data_dir: Path | str = DEFAULT_DIR) -> Path:
    """Download the SemEval-2014 gold test file for ``domain`` (cached)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    url = SEMEVAL_URLS[domain]
    path = data_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL
    return path


def load_semeval(path: Path | str) -> list[GoldSentence]:
    """Parse a SemEval-2014 Task 4 XML file into ``GoldSentence`` objects."""
    root = ET.parse(path).getroot()
    sentences = []
    for s in root.iter("sentence"):
        aspects = [
            GoldAspect(
                term=t.get("term", ""),
                polarity=t.get("polarity", "neutral"),
                span=(int(t.get("from", 0)), int(t.get("to", 0))),
            )
            for t in s.iter("aspectTerm")
        ]
        sentences.append(GoldSentence(id=s.get("id", ""), text=s.findtext("text", ""), aspects=aspects))
    return sentences
