"""PCA-truncated LLM item-embedding initialiser for SASRec.item_emb.

Used by R_llm_init (see briefs/r_llm_init.md): a representation-transfer
control that replaces SASRec's trunc_normal(std=0.02) item_emb init with a
PCA projection of cached Qwen2-7B per-item embeddings. Supervised-only
training with this init isolates whether the RL-over-SL ranking lift is
driven by reward signal vs LLM representation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def fit_pca_item_init(cache_npz_path: Path | str,
                      item_index_parquet: Path | str,
                      num_items: int,
                      k: int = 128,
                      target_std: float = 0.02,
                      seed: int = 0) -> torch.Tensor:
    """Build a (num_items, k) init tensor from a Qwen item-embedding cache.

    - Fits PCA on the item-vocab subset of the cache (float64 centred, SVD).
    - Projects to k components; row-wise scale is renormalised so the
      elementwise std of the covered rows equals ``target_std``.
    - Missing items fall back to ``trunc_normal``-analogue random with the
      same target_std (count is logged).
    - Row 0 (padding_idx) is zeroed after projection.

    Returns float32 torch tensor on CPU; caller copies into nn.Embedding.
    """
    cache_npz_path = Path(cache_npz_path)
    item_index_parquet = Path(item_index_parquet)

    z = np.load(cache_npz_path)
    item_index = pd.read_parquet(item_index_parquet)
    raw_to_idx = {str(r): int(i) for r, i in
                  zip(item_index["raw_item_id"].astype(str),
                      item_index["item_idx"])}

    d = None
    covered: list[int] = []
    missing: list[int] = []
    vecs: list[np.ndarray] = []
    for raw, idx in raw_to_idx.items():
        if raw in z.files:
            v = np.asarray(z[raw], dtype=np.float64).ravel()
            if d is None:
                d = v.shape[0]
            if v.shape[0] == d:
                covered.append(idx)
                vecs.append(v)
                continue
        missing.append(idx)

    if d is None:
        raise RuntimeError(
            f"[fit_pca_item_init] no cache keys matched any raw_item_id in "
            f"{item_index_parquet}; cache has {len(z.files)} keys, "
            f"item_index has {len(raw_to_idx)} rows.")

    X = np.stack(vecs, axis=0)                       # (n_cov, d) float64
    X -= X.mean(axis=0, keepdims=True)

    # SVD-based PCA (deterministic; no sklearn dep).
    # X ~ U @ diag(S) @ Vt;  principal scores P_full = X @ V = U * S.
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    k_eff = min(k, Vt.shape[0])
    P = (X @ Vt[:k_eff].T).astype(np.float32)        # (n_cov, k_eff)
    # If k_eff < k, right-pad with zeros so the output shape is exact.
    if k_eff < k:
        pad = np.zeros((P.shape[0], k - k_eff), dtype=np.float32)
        P = np.concatenate([P, pad], axis=1)

    evr = float((S[:k_eff] ** 2).sum() / (S ** 2).sum()) if S.size else 0.0

    # --- renormalisation (covered rows): shared scalar rescale to target_std ---
    # A shared scalar preserves inter-row variance structure (popularity /
    # semantic signal carried by the PCs); a per-row rescale would destroy it.
    s_elem = float(P.std())
    if s_elem > 0:
        P *= (target_std / s_elem)

    init = np.zeros((num_items, k), dtype=np.float32)
    for row, idx in enumerate(covered):
        if 0 <= idx < num_items:
            init[idx] = P[row]

    # --- fallback: trunc_normal-analogue draws, per-row rescaled ---
    rng = np.random.default_rng(seed)
    for idx in missing:
        if 0 <= idx < num_items:
            v = rng.standard_normal(k).astype(np.float32)
            s = float(v.std())
            if s > 0:
                v *= (target_std / s)
            init[idx] = v

    # padding_idx=0 stays zero.
    init[0] = 0.0

    n_cov = sum(1 for idx in covered if 0 <= idx < num_items)
    n_miss = sum(1 for idx in missing if 0 <= idx < num_items)
    ds_tag = cache_npz_path.parent.parent.name  # .../cache/{ds}/qwen2_7b/x.npz
    print(f"[R_llm_init] PCA k={k} explained_variance_ratio={evr:.4f} "
          f"(ds={ds_tag}, n_items_covered={n_cov}/{num_items - 1}, "
          f"n_missing_fallback={n_miss})")

    return torch.from_numpy(init)
