"""Dataset loaders: parse raw interactions, build user/item maps and user sequences.

Outputs a pickle per dataset:
    {
      'dataset': 'ml1m',
      'user2idx': {str/int: int},
      'item2idx': {str/int: int},
      'idx2user': list,
      'idx2item': list,
      'sequences':  list-of-list-of-(item_idx, y, ts)  sorted by ts,
      'n_users': int,
      'n_items': int,
      'splits': {'train':[list of (iid,y,ts)], 'val':(iid,y,ts), 'test':(iid,y,ts)} per user
    }
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.config import dataset_config, project_root


# ---------------------------------------------------------------------------
# Per-dataset raw parsers
# ---------------------------------------------------------------------------

def _load_ml1m(cfg: dict) -> pd.DataFrame:
    path = project_root() / cfg["paths"]["ratings"]
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["user", "item", "rating", "ts"],
        engine="python",
        encoding="latin-1",
    )
    df["user"] = df["user"].astype(str)
    df["item"] = df["item"].astype(str)
    df["ts"] = df["ts"].astype(np.int64)
    return df


def _load_videogames(cfg: dict) -> pd.DataFrame:
    path = project_root() / cfg["paths"]["ratings"]
    # CSV: asin, reviewerID, rating, ts
    df = pd.read_csv(path, header=None, names=["item", "user", "rating", "ts"])
    df["user"] = df["user"].astype(str)
    df["item"] = df["item"].astype(str)
    df["ts"] = df["ts"].astype(np.int64)
    return df


def _load_yelp(cfg: dict) -> pd.DataFrame:
    path = project_root() / cfg["paths"]["ratings"]
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                dict(
                    user=r["user_id"],
                    item=r["business_id"],
                    rating=float(r["stars"]),
                    ts=int(
                        pd.Timestamp(r.get("date", "1970-01-01")).value // 10**9
                    ),
                )
            )
    df = pd.DataFrame(rows)
    return df


_LOADERS = {
    "ml1m": _load_ml1m,
    "videogames": _load_videogames,
    "yelp": _load_yelp,
}


# ---------------------------------------------------------------------------
# Sequence building + splits
# ---------------------------------------------------------------------------

def _build_maps(df: pd.DataFrame, item_cache_ids: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """item2idx uses the FULL keyword-cache ID universe (so the agent can recommend
    any cached item, including cold-start ones). Ratings for items outside the cache
    are dropped.
    """
    item_set = set(item_cache_ids)
    df_in = df[df["item"].isin(item_set)]
    user_ids = sorted(df_in["user"].unique().tolist())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_cache_ids)}
    return user2idx, item2idx


def _load_item_cache_ids(cfg: dict) -> List[str]:
    """Ordered unique item IDs from the keywords_cache json (deduplicated, first-wins)."""
    path = project_root() / cfg["paths"]["keywords_cache"]
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    id_field = cfg["id_field"]
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        sid = str(it[id_field])
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _leave_last_out(
    df: pd.DataFrame,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    rating_threshold: int,
) -> dict:
    """Returns:
        sequences: list[n_users] of list[(item_idx, y, ts)]
        train/val/test per user (indices into the sorted sequence)
    """
    df = df[df["user"].isin(user2idx) & df["item"].isin(item2idx)].copy()
    df["uidx"] = df["user"].map(user2idx)
    df["iidx"] = df["item"].map(item2idx)
    df["y"] = (df["rating"] >= rating_threshold).astype(np.int8)
    df = df.sort_values(["uidx", "ts", "iidx"]).reset_index(drop=True)

    n_users = len(user2idx)
    sequences: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_users)]
    for uidx, group in df.groupby("uidx", sort=False):
        sequences[uidx] = list(
            zip(
                group["iidx"].tolist(),
                group["y"].tolist(),
                group["ts"].tolist(),
            )
        )

    # Train/val/test per user
    train: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_users)]
    val: List[Tuple[int, int, int] | None] = [None] * n_users
    test: List[Tuple[int, int, int] | None] = [None] * n_users
    for u in range(n_users):
        seq = sequences[u]
        if len(seq) >= 3:
            train[u] = seq[:-2]
            val[u] = seq[-2]
            test[u] = seq[-1]
        elif len(seq) == 2:
            train[u] = seq[:-1]
            test[u] = seq[-1]
        else:
            train[u] = seq
    return {"sequences": sequences, "train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_splits(dataset: str) -> dict:
    cfg = dataset_config(dataset)
    print(f"[{dataset}] loading raw ratings ...")
    df = _LOADERS[dataset](cfg)
    print(f"[{dataset}] raw rows: {len(df):,}")

    item_cache_ids = _load_item_cache_ids(cfg)
    print(f"[{dataset}] items in cache: {len(item_cache_ids):,}")

    user2idx, item2idx = _build_maps(df, item_cache_ids)
    print(f"[{dataset}] users: {len(user2idx):,}  items(used): {len(item2idx):,}")

    splits = _leave_last_out(df, user2idx, item2idx, cfg["rating_threshold"])
    n_train = sum(len(s) for s in splits["train"])
    print(f"[{dataset}] train interactions: {n_train:,}")

    payload = dict(
        dataset=dataset,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2user=list(user2idx.keys()),
        idx2item=list(item2idx.keys()),
        sequences=splits["sequences"],
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        n_users=len(user2idx),
        n_items=len(item2idx),
    )

    out_path = project_root() / cfg["paths"]["split_pkl"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[{dataset}] wrote {out_path}")
    return payload


def load_splits(dataset: str) -> dict:
    cfg = dataset_config(dataset)
    path = project_root() / cfg["paths"]["split_pkl"]
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(_LOADERS))
    args = ap.parse_args()
    build_splits(args.dataset)


if __name__ == "__main__":
    main()
