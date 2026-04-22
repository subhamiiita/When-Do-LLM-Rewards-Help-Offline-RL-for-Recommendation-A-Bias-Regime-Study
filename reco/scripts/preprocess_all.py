"""Run preprocessing for all three datasets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import run

for ds in ("movielens-1m", "amazon-videogames", "yelp"):
    try:
        run(ds)
    except Exception as e:
        print(f"[skip {ds}] {type(e).__name__}: {e}")
