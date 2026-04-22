"""
Unify all three datasets into a canonical parquet schema:
    user_idx (int32), item_idx (int32), rating (float32), ts (int64)

Also writes:
    item_index.parquet   (item_idx, raw_item_id)  — string id -> contiguous int
    user_index.parquet   (user_idx, raw_user_id)
    meta.json            (num_users, num_items, dataset name, stats)

Item_idx 0 is reserved as PAD. Real items start at 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "processed"


# ---------- raw readers ----------

def _read_ml1m() -> pd.DataFrame:
    path = DATA_DIR / "movielens-1m" / "ratings.dat"
    rows = []
    with open(path, "rb") as f:
        for line in f:
            parts = line.decode("latin-1").strip().split("::")
            if len(parts) != 4:
                continue
            u, i, r, t = parts
            rows.append((u, i, float(r), int(t)))
    df = pd.DataFrame(rows, columns=["raw_user", "raw_item", "rating", "ts"])
    return df


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read_vg() -> pd.DataFrame:
    # CSV has columns: user_id, item_id (asin), rating, timestamp (typical Amazon layout)
    path = DATA_DIR / "amazon-videogames" / "Video_Games_10core.csv"
    try:
        df = pd.read_csv(path, header=None,
                         names=["raw_item", "raw_user", "rating", "ts"])
    except Exception:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            cols.get("user_id", "raw_user"): "raw_user",
            cols.get("asin", "raw_item"): "raw_item",
            cols.get("rating", "rating"): "rating",
            cols.get("timestamp", "ts"): "ts",
        })[["raw_user", "raw_item", "rating", "ts"]]
    df["rating"] = df["rating"].astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df


def _read_yelp() -> pd.DataFrame:
    path = DATA_DIR / "yelp" / "mo_review_10core.json"
    rows = []
    for obj in _iter_jsonl(path):
        try:
            u = obj["user_id"]
            i = obj["business_id"]
            r = float(obj.get("stars", obj.get("rating", 0)))
            t_raw = obj.get("date") or obj.get("ts") or obj.get("timestamp")
            if isinstance(t_raw, str):
                # e.g. "2016-03-14 22:22:33"
                ts = int(pd.Timestamp(t_raw).timestamp())
            else:
                ts = int(t_raw or 0)
            rows.append((u, i, r, ts))
        except KeyError:
            continue
    return pd.DataFrame(rows, columns=["raw_user", "raw_item", "rating", "ts"])


READERS = {
    "movielens-1m": _read_ml1m,
    "amazon-videogames": _read_vg,
    "yelp": _read_yelp,
}


# ---------- canonicalise ----------

def canonicalise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.dropna().drop_duplicates(subset=["raw_user", "raw_item"], keep="last")
    df = df.sort_values(["raw_user", "ts"], kind="stable").reset_index(drop=True)

    # user_idx: 0..num_users-1 (dense)
    user_ids = df["raw_user"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    # item_idx: 1..num_items (0 = PAD)
    item_ids = df["raw_item"].unique()
    item_map = {it: i + 1 for i, it in enumerate(item_ids)}

    df["user_idx"] = df["raw_user"].map(user_map).astype("int32")
    df["item_idx"] = df["raw_item"].map(item_map).astype("int32")
    df["rating"] = df["rating"].astype("float32")
    df["ts"] = df["ts"].astype("int64")

    user_index = pd.DataFrame({
        "user_idx": np.arange(len(user_ids), dtype="int32"),
        "raw_user_id": user_ids,
    })
    item_index = pd.DataFrame({
        "item_idx": np.arange(1, len(item_ids) + 1, dtype="int32"),
        "raw_item_id": item_ids,
    })
    df = df[["user_idx", "item_idx", "rating", "ts"]]
    return df, user_index, item_index


def run(dataset: str) -> None:
    if dataset not in READERS:
        raise ValueError(f"unknown dataset: {dataset}")
    out = OUT_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] reading {dataset} ...")
    raw = READERS[dataset]()
    print(f"[preprocess] raw rows: {len(raw):,}")
    df, users, items = canonicalise(raw)

    df.to_parquet(out / "interactions.parquet", index=False)
    users.to_parquet(out / "user_index.parquet", index=False)
    items.to_parquet(out / "item_index.parquet", index=False)

    # user-length distribution
    lens = df.groupby("user_idx").size()
    meta = {
        "dataset": dataset,
        "num_users": int(len(users)),
        "num_items": int(len(items)),
        "num_interactions": int(len(df)),
        "avg_user_len": float(lens.mean()),
        "p50_user_len": float(lens.median()),
        "p95_user_len": float(lens.quantile(0.95)),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(READERS.keys()))
    args = ap.parse_args()
    run(args.dataset)
