"""Load UG-MORS extraction artifacts into tensor-friendly structures.

For a dataset, we build:
    item_emb:        (V, D)  — nomic-embed-text-v1.5, L2-normalized
    u_jml, u_sem, u_nli, u_llm: (V,) per-item uncertainty components
    pos_kw_ids:      (V, K)  — top-K positive keyword ids (pad=0)
    pos_kw_conf:     (V, K)  — positive keyword confidences (0 where pad)
    neg_kw_ids/conf: (V, K)  — same, negative
    kw_emb:          (Nk, D) — keyword embedding table (row 0 = PAD zero)

We map keyword strings -> ids via a single dictionary across dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
PROC_DIR = ROOT / "processed"


_DATASET_FILES = {
    "movielens-1m": {
        "kw":   "keywords_cache_ml1m.json",
        "ie":   "item_embeddings_ml1m.npz",
        "ke":   "keyword_embeddings_ml1m.npz",
        "id_field": "movie_id",
    },
    "amazon-videogames": {
        "kw":   "keywords_cache.json",
        "ie":   "item_embeddings.npz",
        "ke":   "keyword_embeddings.npz",
        "id_field": "asin",
    },
    "yelp": {
        "kw":   "keywords_cache.json",
        "ie":   "item_embeddings.npz",
        "ke":   "keyword_embeddings.npz",
        "id_field": "business_id",
    },
}


@dataclass
class SimulatorCache:
    dataset: str
    num_items: int            # includes PAD (idx 0)
    emb_dim: int
    item_emb: np.ndarray      # (V, D) float32, row 0 = zeros
    u_jml: np.ndarray         # (V,)
    u_sem: np.ndarray
    u_nli: np.ndarray
    u_llm: np.ndarray
    pos_kw_ids: np.ndarray    # (V, K) int32  — 0 = pad
    pos_kw_conf: np.ndarray   # (V, K) float32
    neg_kw_ids: np.ndarray    # (V, K)
    neg_kw_conf: np.ndarray
    kw_emb: np.ndarray        # (Nk+1, D), row 0 = zeros
    raw_to_idx: Dict[str, int]   # raw item id -> item_idx (from preprocess)


def _load_raw_to_idx(dataset: str) -> Dict[str, int]:
    import pandas as pd
    idx = pd.read_parquet(PROC_DIR / dataset / "item_index.parquet")
    return {str(r): int(i) for r, i in zip(idx["raw_item_id"], idx["item_idx"])}


def _avg_soft(runs, side: str) -> Dict[str, float]:
    """Average per-keyword confidence across self-consistency runs."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    M = max(1, len(runs))
    for run in runs or []:
        for obj in run.get(f"{side}_soft", []) or []:
            kw = obj.get("keyword")
            if not kw:
                continue
            c = float(obj.get("confidence", 0.0))
            sums[kw] = sums.get(kw, 0.0) + c
            counts[kw] = counts.get(kw, 0) + 1
    # expected confidence (marginal over runs): sum / M so keywords appearing
    # in only a subset of runs are penalised — supports the self-consistency idea
    return {kw: s / M for kw, s in sums.items()}


def build_cache(dataset: str, topk_keywords: int = 12) -> SimulatorCache:
    cfg = _DATASET_FILES[dataset]
    cache_dir = CACHE_DIR / dataset / "qwen2_7b"
    with open(cache_dir / cfg["kw"], "r", encoding="utf-8") as f:
        items = json.load(f)
    # normalise to list
    if isinstance(items, dict):
        items = list(items.values())

    raw_to_idx = _load_raw_to_idx(dataset)

    ie = np.load(cache_dir / cfg["ie"])
    ke = np.load(cache_dir / cfg["ke"])

    # build keyword vocab
    kw_list = list(ke.files)
    kw_to_id = {kw: i + 1 for i, kw in enumerate(kw_list)}  # 0 = PAD
    D = ke[kw_list[0]].shape[0] if kw_list else ie[ie.files[0]].shape[0]

    Nk = len(kw_list)
    kw_emb = np.zeros((Nk + 1, D), dtype=np.float32)
    for kw, i in kw_to_id.items():
        kw_emb[i] = ke[kw].astype(np.float32)

    V = max(raw_to_idx.values()) + 1 if raw_to_idx else 1
    item_emb = np.zeros((V, D), dtype=np.float32)
    u_jml = np.zeros(V, dtype=np.float32)
    u_sem = np.zeros(V, dtype=np.float32)
    u_nli = np.zeros(V, dtype=np.float32)
    u_llm = np.zeros(V, dtype=np.float32)
    pos_ids  = np.zeros((V, topk_keywords), dtype=np.int32)
    pos_conf = np.zeros((V, topk_keywords), dtype=np.float32)
    neg_ids  = np.zeros((V, topk_keywords), dtype=np.int32)
    neg_conf = np.zeros((V, topk_keywords), dtype=np.float32)

    id_field = cfg["id_field"]
    hits = 0
    for obj in items:
        raw = obj.get(id_field)
        if raw is None:
            continue
        raw_s = str(raw)
        idx = raw_to_idx.get(raw_s)
        if idx is None:
            continue
        hits += 1
        # item embedding
        if raw_s in ie.files:
            item_emb[idx] = ie[raw_s].astype(np.float32)
        unc = obj.get("uncertainty", {}) or {}
        u_jml[idx] = float(unc.get("u_jml", 0.0))
        u_sem[idx] = float(unc.get("u_sem", 0.0))
        u_nli[idx] = float(unc.get("u_nli", 0.0))
        u_llm[idx] = float(unc.get("u_llm", 0.0))

        # soft keyword confidences per side
        pos_soft = _avg_soft(obj.get("runs") or [], "pos")
        neg_soft = _avg_soft(obj.get("runs") or [], "neg")
        if not pos_soft and obj.get("pos"):
            pos_soft = {kw: 1.0 for kw in obj["pos"]}
        if not neg_soft and obj.get("neg"):
            neg_soft = {kw: 1.0 for kw in obj["neg"]}

        for arr_ids, arr_conf, soft in [(pos_ids, pos_conf, pos_soft),
                                         (neg_ids, neg_conf, neg_soft)]:
            ranked = sorted(soft.items(), key=lambda kv: -kv[1])[:topk_keywords]
            for slot, (kw, c) in enumerate(ranked):
                kid = kw_to_id.get(kw, 0)
                arr_ids[idx, slot] = kid
                arr_conf[idx, slot] = c if kid > 0 else 0.0

    return SimulatorCache(
        dataset=dataset, num_items=V, emb_dim=D,
        item_emb=item_emb,
        u_jml=u_jml, u_sem=u_sem, u_nli=u_nli, u_llm=u_llm,
        pos_kw_ids=pos_ids, pos_kw_conf=pos_conf,
        neg_kw_ids=neg_ids, neg_kw_conf=neg_conf,
        kw_emb=kw_emb,
        raw_to_idx=raw_to_idx,
    )
