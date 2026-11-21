"""Build the Yelp review sample used by the notebooks.

The Yelp Open Dataset cannot be redistributed, so it is not in this repo.
Download it yourself (free, requires accepting Yelp's terms):

- https://www.yelp.com/dataset, or
- https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

then point this script at ``yelp_academic_dataset_review.json``:

    python scripts/prepare_yelp.py path/to/yelp_academic_dataset_review.json

It writes ``data/yelp/yelp_sample.json`` (JSON Lines with ``text`` and
``stars``), a uniform random sample drawn with a fixed seed so it is
reproducible. Reservoir sampling streams the ~5 GB file without loading it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path("data/yelp/yelp_sample.json")


def sample_reviews(path: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            review = json.loads(line)
            row = {"text": review["text"], "stars": float(review["stars"])}
            if i < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = row
    return reservoir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_file", type=Path, help="yelp_academic_dataset_review.json")
    ap.add_argument("-n", type=int, default=20_000, help="number of reviews to sample (default 20000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = sample_reviews(args.review_file, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} reviews to {args.out}")


if __name__ == "__main__":
    main()
