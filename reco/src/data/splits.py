"""Leave-last-out split with a calibration holdout for conformal gate.

For each user (needs >= 3 interactions):
    - last interaction  -> TEST
    - second-to-last    -> VAL
    - all earlier       -> TRAIN

A `val_frac` fraction of *users* is additionally held out as a CALIBRATION
set for the conformal UG-MORS gate. Those users' last interaction is used
as the real-rating reference; their earlier history still goes into TRAIN.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    calib_users: np.ndarray  # user_idxs reserved for CRC calibration
    num_users: int
    num_items: int


def make_splits(interactions_parquet: str | Path,
                val_frac: float = 0.1,
                seed: int = 42) -> Splits:
    df = pd.read_parquet(interactions_parquet)
    df = df.sort_values(["user_idx", "ts"], kind="stable").reset_index(drop=True)

    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["item_idx"].max()) + 1  # 0 = PAD included

    df["rank"] = df.groupby("user_idx").cumcount()
    user_lens = df.groupby("user_idx").size()
    good_users = user_lens[user_lens >= 3].index
    df = df[df["user_idx"].isin(good_users)].reset_index(drop=True)

    df["rev_rank"] = df.groupby("user_idx").cumcount(ascending=False)
    test = df[df["rev_rank"] == 0].drop(columns=["rank", "rev_rank"])
    val = df[df["rev_rank"] == 1].drop(columns=["rank", "rev_rank"])
    train = df[df["rev_rank"] >= 2].drop(columns=["rank", "rev_rank"])

    rng = np.random.default_rng(seed)
    all_u = np.sort(good_users.values)
    idx = rng.permutation(len(all_u))
    n_calib = int(round(val_frac * len(all_u)))
    calib_users = np.sort(all_u[idx[:n_calib]])

    return Splits(train=train.reset_index(drop=True),
                  val=val.reset_index(drop=True),
                  test=test.reset_index(drop=True),
                  calib_users=calib_users,
                  num_users=num_users,
                  num_items=num_items)


def user_sequences(train: pd.DataFrame) -> Dict[int, List[Tuple[int, float, int]]]:
    out: Dict[int, List[Tuple[int, float, int]]] = {}
    for u, g in train.sort_values(["user_idx", "ts"]).groupby("user_idx"):
        out[int(u)] = list(zip(g["item_idx"].astype(int).tolist(),
                                g["rating"].astype(float).tolist(),
                                g["ts"].astype("int64").tolist()))
    return out
