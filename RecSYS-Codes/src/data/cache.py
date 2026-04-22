"""Load keyword/embedding caches and precompute per-item pooled keyword vectors (E_pos, E_neg).

Returns a ``CacheBundle`` with GPU-resident tensors used by reward modules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device


@dataclass
class CacheBundle:
    dataset: str
    n_items: int
    item_emb: torch.Tensor          # (n_items, 768) fp16 on device, L2-normalized
    E_pos: torch.Tensor             # (n_items, 768) pooled, L2-normalized (zero row if empty)
    E_neg: torch.Tensor             # (n_items, 768)
    pos_kw_ids: torch.Tensor        # (n_items, MAX_POS) padded with -1
    neg_kw_ids: torch.Tensor        # (n_items, MAX_NEG)
    pos_kw_count: torch.Tensor      # (n_items,) long
    neg_kw_count: torch.Tensor      # (n_items,) long
    u_llm: torch.Tensor             # (n_items,) float32 on device
    u_jml: torch.Tensor
    u_sem: torch.Tensor
    u_nli: torch.Tensor
    n_keywords: int                 # size of shared keyword vocabulary
    item_categories: List[str]      # primary category per item (for h_C)
    item_titles: List[str]          # for persona text
    items_raw: list                 # the raw list loaded from keywords_cache.json


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def build_cache(dataset: str, device: Optional[torch.device] = None) -> CacheBundle:
    cfg = dataset_config(dataset)
    device = device or get_device()

    # ---- items json (authoritative ordering, deduped first-wins) ----
    with (project_root() / cfg["paths"]["keywords_cache"]).open("r", encoding="utf-8") as f:
        raw_items = json.load(f)
    id_field = cfg["id_field"]
    seen: set[str] = set()
    items: list = []
    id_strs: List[str] = []
    for it in raw_items:
        sid = str(it[id_field])
        if sid in seen:
            continue
        seen.add(sid)
        items.append(it)
        id_strs.append(sid)
    n_items = len(items)

    # ---- item embeddings ----
    ie = np.load(project_root() / cfg["paths"]["item_emb_npz"])
    # Some datasets (e.g. videogames) have MORE emb keys than items - only keep ones aligned to the cache order.
    item_emb = np.zeros((n_items, 768), dtype=np.float32)
    miss = 0
    for i, k in enumerate(id_strs):
        if k in ie:
            item_emb[i] = ie[k]
        else:
            miss += 1
    if miss:
        print(f"[cache {dataset}] WARNING: {miss} items lacked embeddings; zero vectors used")
    item_emb_t = _l2_normalize(torch.from_numpy(item_emb).to(device=device, dtype=torch.float32))

    # ---- keyword embeddings / shared vocabulary ----
    ke = np.load(project_root() / cfg["paths"]["keyword_emb_npz"])
    # Build keyword vocabulary from the UNION of per-item pos & neg keyword lists (strings).
    kw2idx: Dict[str, int] = {}
    for it in items:
        for k in it.get("pos", []):
            if k not in kw2idx:
                kw2idx[k] = len(kw2idx)
        for k in it.get("neg", []):
            if k not in kw2idx:
                kw2idx[k] = len(kw2idx)
    n_keywords = len(kw2idx)

    # Build a (n_keywords, 768) matrix. Words missing from kw embedding npz get a zero row.
    kw_emb = np.zeros((n_keywords, 768), dtype=np.float32)
    miss_kw = 0
    for k, idx in kw2idx.items():
        if k in ke:
            kw_emb[idx] = ke[k]
        else:
            miss_kw += 1
    if miss_kw:
        print(f"[cache {dataset}] WARNING: {miss_kw} keywords lacked embeddings")
    kw_emb_t = _l2_normalize(torch.from_numpy(kw_emb).to(device=device, dtype=torch.float32))

    # ---- padded per-item kw id matrices + pooled E_pos/E_neg ----
    max_pos = max((len(it.get("pos", [])) for it in items), default=0)
    max_neg = max((len(it.get("neg", [])) for it in items), default=0)
    max_pos = max(max_pos, 1)
    max_neg = max(max_neg, 1)
    pos_kw_ids = -np.ones((n_items, max_pos), dtype=np.int64)
    neg_kw_ids = -np.ones((n_items, max_neg), dtype=np.int64)
    pos_count = np.zeros(n_items, dtype=np.int64)
    neg_count = np.zeros(n_items, dtype=np.int64)
    E_pos = np.zeros((n_items, 768), dtype=np.float32)
    E_neg = np.zeros((n_items, 768), dtype=np.float32)
    for i, it in enumerate(items):
        p = it.get("pos", [])
        n = it.get("neg", [])
        if p:
            idxs = [kw2idx[k] for k in p]
            pos_kw_ids[i, : len(idxs)] = idxs
            pos_count[i] = len(idxs)
            E_pos[i] = kw_emb[idxs].mean(axis=0)
        if n:
            idxs = [kw2idx[k] for k in n]
            neg_kw_ids[i, : len(idxs)] = idxs
            neg_count[i] = len(idxs)
            E_neg[i] = kw_emb[idxs].mean(axis=0)

    E_pos_t = _l2_normalize(torch.from_numpy(E_pos).to(device=device, dtype=torch.float32))
    E_neg_t = _l2_normalize(torch.from_numpy(E_neg).to(device=device, dtype=torch.float32))
    pos_kw_ids_t = torch.from_numpy(pos_kw_ids).to(device=device)
    neg_kw_ids_t = torch.from_numpy(neg_kw_ids).to(device=device)
    pos_count_t = torch.from_numpy(pos_count).to(device=device)
    neg_count_t = torch.from_numpy(neg_count).to(device=device)

    # ---- uncertainty ----
    u_llm = np.array([float(it["uncertainty"].get("u_llm", 0.5)) for it in items], dtype=np.float32)
    u_jml = np.array([float(it["uncertainty"].get("u_jml", 0.5)) for it in items], dtype=np.float32)
    u_sem = np.array([float(it["uncertainty"].get("u_sem", 0.0)) for it in items], dtype=np.float32)
    u_nli = np.array([float(it["uncertainty"].get("u_nli", 0.0)) for it in items], dtype=np.float32)

    # ---- primary category (for h_C) + title ----
    cat_field = cfg["category_field"]
    cat_sep = cfg["category_sep"]
    cats: List[str] = []
    titles: List[str] = []
    for it in items:
        raw = it.get(cat_field, "")
        if isinstance(raw, list):
            raw = cat_sep.join(raw) if raw else ""
        first = raw.split(cat_sep)[0].strip().lower() if raw else ""
        cats.append(first)
        titles.append(str(it.get("title") or it.get("name") or ""))

    return CacheBundle(
        dataset=dataset,
        n_items=n_items,
        item_emb=item_emb_t,
        E_pos=E_pos_t,
        E_neg=E_neg_t,
        pos_kw_ids=pos_kw_ids_t,
        neg_kw_ids=neg_kw_ids_t,
        pos_kw_count=pos_count_t,
        neg_kw_count=neg_count_t,
        u_llm=torch.from_numpy(u_llm).to(device),
        u_jml=torch.from_numpy(u_jml).to(device),
        u_sem=torch.from_numpy(u_sem).to(device),
        u_nli=torch.from_numpy(u_nli).to(device),
        n_keywords=n_keywords,
        item_categories=cats,
        item_titles=titles,
        items_raw=items,
    )


def build_keyword_indicator(cache: CacheBundle) -> torch.Tensor:
    """Sparse (n_items, n_keywords) binary indicator telling us which keywords each
    item has marked in EITHER pos or neg list. Useful for set-overlap cardinality.

    We return a DENSE fp16 tensor here; for n_items<=5K and n_keywords<=40K, this is
    ~400 MB fp16 — within 12GB budget. For larger datasets we'd switch to sparse.
    """
    device = cache.item_emb.device
    n_items = cache.n_items
    n_kw = cache.n_keywords
    pos_ind = torch.zeros((n_items, n_kw), dtype=torch.float16, device=device)
    neg_ind = torch.zeros((n_items, n_kw), dtype=torch.float16, device=device)
    pos = cache.pos_kw_ids
    neg = cache.neg_kw_ids
    for i in range(n_items):
        p = pos[i][pos[i] >= 0]
        n = neg[i][neg[i] >= 0]
        if p.numel():
            pos_ind[i, p] = 1.0
        if n.numel():
            neg_ind[i, n] = 1.0
    return pos_ind, neg_ind


def build_keyword_indicator_fast(cache: CacheBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized version that avoids the Python loop via scatter_ on padded ids."""
    device = cache.item_emb.device
    n_items = cache.n_items
    n_kw = cache.n_keywords
    # Map -1 padding to n_kw (an extra trailing bin we'll slice off)
    pos = cache.pos_kw_ids.clone()
    neg = cache.neg_kw_ids.clone()
    pos[pos < 0] = n_kw
    neg[neg < 0] = n_kw
    pos_ind = torch.zeros((n_items, n_kw + 1), dtype=torch.float16, device=device)
    neg_ind = torch.zeros((n_items, n_kw + 1), dtype=torch.float16, device=device)
    pos_ind.scatter_(1, pos, torch.ones_like(pos, dtype=torch.float16))
    neg_ind.scatter_(1, neg, torch.ones_like(neg, dtype=torch.float16))
    # Drop the trailing padding column
    return pos_ind[:, :n_kw].contiguous(), neg_ind[:, :n_kw].contiguous()
