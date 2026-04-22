"""Precompute per-(user, item) NLI scores for the top-K SASRec-candidate items per user.

Stores a dense (n_users, n_items) float16 matrix in [-1, 1]; untouched pairs default to 0.0.
Also saves a boolean mask telling which pairs were actually computed.

--smoke runs on 100 users and top-20 candidates to verify output distribution quickly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.common.seed import set_seed
from src.data.cache import build_cache
from src.data.loaders import load_splits
from src.data.persona import build_item_descriptions, build_user_personas
from src.models.nli_persona import NLIScorer
from src.models.sasrec import SASRec, SASRecConfig


def top_k_candidates_per_user(
    splits: dict, sas_ckpt: Path, device: torch.device, max_len: int, top_k: int
) -> np.ndarray:
    ck = torch.load(sas_ckpt, map_location=device, weights_only=False)
    mcfg = SASRecConfig(**ck["cfg"])
    model = SASRec(mcfg).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    n_users = splits["n_users"]
    n_items = splits["n_items"]
    pad = model.pad_id
    top = np.zeros((n_users, top_k), dtype=np.int64)

    BATCH = 256
    u = 0
    while u < n_users:
        batch_u = list(range(u, min(u + BATCH, n_users)))
        seqs = np.full((len(batch_u), max_len), pad, dtype=np.int64)
        seen_masks = []
        for i, uu in enumerate(batch_u):
            hist = [it for (it, _, _) in splits["train"][uu]]
            take = hist[-max_len:]
            if take:
                seqs[i, -len(take):] = take
            seen_masks.append(set(it for (it, _, _) in splits["sequences"][uu]))
        seq_t = torch.from_numpy(seqs).to(device)
        with torch.no_grad():
            h = model.last_hidden(seq_t)
            scores = model.score_all(h).float().cpu().numpy()
        for i, uu in enumerate(batch_u):
            s = scores[i].copy()
            for iid in seen_masks[i]:
                if 0 <= iid < n_items:
                    s[iid] = -1e9
            order = np.argpartition(-s, top_k)[:top_k]
            order = order[np.argsort(-s[order])]
            top[uu] = order
        u += BATCH
    return top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Only process 100 users × top-20 candidates for quick validation")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = dataset_config(args.dataset)
    device = get_device()
    splits = load_splits(args.dataset)
    cache = build_cache(args.dataset, device)

    sa = cfg["sasrec"]
    nli_cfg = cfg["nli"]
    sas_ckpt = project_root() / cfg["paths"]["sasrec_ckpt"]

    n_users_limit = 100 if args.smoke else splits["n_users"]
    top_k_used = 20 if args.smoke else nli_cfg["top_k_candidates"]

    print(f"[nli {args.dataset}] Building top-{top_k_used} SASRec candidates per user ...")
    t0 = time.time()
    top = top_k_candidates_per_user(splits, sas_ckpt, device, sa["max_len"], top_k_used)
    print(f"[nli {args.dataset}] top-K computed in {time.time()-t0:.0f}s shape={top.shape}")

    print(f"[nli {args.dataset}] Building personas (top_cats={nli_cfg['persona_top_cats']}, top_kws={nli_cfg['persona_top_kws']}) ...")
    personas = build_user_personas(
        splits, cache, top_cats=nli_cfg["persona_top_cats"], top_kws=nli_cfg["persona_top_kws"]
    )
    descs = build_item_descriptions(cache)

    # --- Final NLI framing: short single-clause hypothesis, rich context premise.
    # premise    = "<persona>. The item in question: <item description>."
    # hypothesis = "This item matches the user's positive preferences."
    # Likes + dislikes both live in the premise; NLI only has to decide one claim.
    HYPOTHESIS = "This item matches the user's positive preferences."
    def build_premise(persona: str, desc: str) -> str:
        return f"{persona.rstrip('.')}. Item in question: {desc}"

    print(f"  persona[0]: {personas[0]!r}")
    print(f"  desc[0]:    {descs[0]!r}")
    print(f"  example premise[0-Item0]: {build_premise(personas[0], descs[0])!r}")
    print(f"  hypothesis (fixed): {HYPOTHESIS!r}")
    print(f"  persona[1]: {personas[min(1, len(personas)-1)]!r}")
    print(f"  desc[1]:    {descs[min(1, len(descs)-1)]!r}")

    print(f"[nli {args.dataset}] Loading NLI model {nli_cfg['model']} ...")
    scorer = NLIScorer(nli_cfg["model"], device)

    n_users = splits["n_users"]
    n_items = splits["n_items"]

    # Build unique (u, iid) pairs to score (only touch the first n_users_limit users).
    seen_set: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for u in range(min(n_users_limit, n_users)):
        for iid in top[u]:
            key = (u, int(iid))
            if key not in seen_set:
                seen_set.add(key)
                pairs.append(key)
        if not args.smoke:
            for (iid, _, _) in splits["train"][u]:
                key = (u, int(iid))
                if key not in seen_set:
                    seen_set.add(key)
                    pairs.append(key)

    print(f"[nli {args.dataset}] Scoring {len(pairs):,} unique pairs ...")

    # Allocate full (n_users, n_items) float16 matrix; untouched cells stay at 0.0 (neutral).
    nli_matrix = np.zeros((n_users, n_items), dtype=np.float16)
    mask = np.zeros((n_users, n_items), dtype=bool)

    CHUNK = 8192
    t0 = time.time()
    for s in range(0, len(pairs), CHUNK):
        chunk = pairs[s : s + CHUNK]
        # premise combines user context + item; hypothesis is a short single-clause claim.
        pr = [build_premise(personas[u], descs[iid]) for (u, iid) in chunk]
        hy = [HYPOTHESIS] * len(chunk)
        scores = scorer.score(pr, hy, batch_size=nli_cfg["batch"], soft=True)  # float32 in [-1, 1]
        for k, (u, iid) in enumerate(chunk):
            nli_matrix[u, iid] = np.float16(scores[k])
            mask[u, iid] = True
        if (s // CHUNK) % 4 == 0:
            el = time.time() - t0
            rate = (s + len(chunk)) / max(1e-9, el)
            eta = (len(pairs) - s - len(chunk)) / max(1e-9, rate)
            print(f"  [{s+len(chunk):,}/{len(pairs):,}]  {rate:.0f} pairs/s  eta={eta/60:.1f}m")

    dt = (time.time() - t0) / 60
    print(f"[nli {args.dataset}] Done scoring in {dt:.1f}m")

    # Distribution summary over computed pairs only
    touched = nli_matrix[mask]
    if len(touched) > 0:
        strong_ent = (touched > 0.5).sum() / len(touched)
        weak_ent = ((touched > 0.1) & (touched <= 0.5)).sum() / len(touched)
        neutral = ((touched >= -0.1) & (touched <= 0.1)).sum() / len(touched)
        weak_con = ((touched < -0.1) & (touched >= -0.5)).sum() / len(touched)
        strong_con = (touched < -0.5).sum() / len(touched)
        print(
            f"[nli {args.dataset}] soft NLI distribution on {len(touched):,} touched pairs:\n"
            f"    strong_entail (>0.5)     : {strong_ent*100:5.2f}%\n"
            f"    weak_entail   (0.1..0.5) : {weak_ent*100:5.2f}%\n"
            f"    neutral       (|x|<=0.1) : {neutral*100:5.2f}%\n"
            f"    weak_contrad  (-0.5..-0.1): {weak_con*100:5.2f}%\n"
            f"    strong_contrad (<-0.5)   : {strong_con*100:5.2f}%\n"
            f"    mean={float(touched.mean()):.3f}  std={float(touched.std()):.3f}"
        )

    out_path = project_root() / cfg["paths"]["nli_cache"]
    if args.smoke:
        out_path = out_path.with_name(out_path.stem + "_smoke.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, nli=nli_matrix, mask=mask)
    print(f"[nli {args.dataset}] saved {out_path}  ({mask.sum():,} computed, {mask.mean()*100:.3f}%)")


if __name__ == "__main__":
    main()
