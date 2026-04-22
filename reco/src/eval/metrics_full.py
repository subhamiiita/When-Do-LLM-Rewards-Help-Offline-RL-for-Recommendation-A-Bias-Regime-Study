"""Full-rank leave-last-out evaluation — the RecSys-acceptable protocol.

Krichene & Rendle (KDD 2020) showed sampled-metric NDCG is NOT a
consistent estimator of full-rank NDCG and can reverse model rankings.
Any serious RecSys submission must report full-rank metrics as the
headline numbers.

Additional metrics supported:
    * Coverage@K       — fraction of items in the catalog ever ranked top-K
    * Diversity@K      — 1 - mean pairwise cosine similarity in the top-K list
    * Novelty@K        — mean -log(p(item)) over the top-K list
    * TailHR@K         — HR@K restricted to users whose true item is in the
                         bottom-80% popularity tail (tests long-tail quality)

All metrics produced together in one pass to amortise the O(|U|*|V|)
scoring cost.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


@torch.no_grad()
def evaluate_ranking_full(agent, enc, splits, hist,
                           topks: List[int] = (1, 5, 10, 20),
                           batch_users: int = 256,
                           device: str = "cuda",
                           item_pop: np.ndarray | None = None,
                           tail_quantile: float = 0.8,
                           max_users: int | None = None,
                           split: str = "test",
                           user_rng_seed: int | None = None) -> Dict[str, float]:
    """Rank the true held-out item against ALL items the user has not seen.

    `split` selects which Splits attribute to evaluate on ("test" | "val").
    `user_rng_seed` makes the max_users subsample deterministic so test and
    val calls within the same epoch score the same user subset.

    Returns a flat dict of metric_name -> value.
    """
    agent.eval()
    eval_df = getattr(splits, split)
    users_all = eval_df["user_idx"].values
    trues_all = eval_df["item_idx"].values
    V = splits.num_items
    L = enc.max_seq_len
    max_k = max(topks)

    if max_users is not None and len(users_all) > max_users:
        if user_rng_seed is not None:
            rng = np.random.default_rng(user_rng_seed)
            idx = rng.choice(len(users_all), size=max_users, replace=False)
        else:
            idx = np.random.choice(len(users_all), size=max_users, replace=False)
        users_all, trues_all = users_all[idx], trues_all[idx]

    # popularity (for novelty + tail metrics)
    if item_pop is None:
        item_pop = np.zeros(V, dtype=np.int64)
        for _, seq in hist.items():
            for it, _, _ in seq:
                if 0 <= it < V:
                    item_pop[it] += 1
    pop_prob = item_pop.astype(np.float64) / max(item_pop.sum(), 1.0)
    pop_prob = np.where(pop_prob > 0, pop_prob, 1.0 / len(pop_prob))

    ranks: List[int] = []
    topk_items_per_user: List[np.ndarray] = []      # for diversity / novelty / coverage
    user_is_tail: List[bool] = []

    # popularity tail threshold
    if item_pop.sum() > 0:
        pop_sorted = np.sort(item_pop)
        tail_cutoff = pop_sorted[int(tail_quantile * len(pop_sorted))]
    else:
        tail_cutoff = float("inf")

    coverage_set: set = set()

    for start in range(0, len(users_all), batch_users):
        end = min(start + batch_users, len(users_all))
        bs = end - start
        u_batch = users_all[start:end]
        t_batch = trues_all[start:end]

        # build sequences (left-padded)
        seqs = np.zeros((bs, L), dtype=np.int64)
        seen_sets = []
        for i, u in enumerate(u_batch):
            items_u = [it for (it, _, _) in hist.get(int(u), [])]
            k = min(L, len(items_u))
            if k > 0:
                seqs[i, -k:] = items_u[-k:]
            seen_sets.append(set(items_u))

        seqs_t = torch.as_tensor(seqs, device=device)
        all_items = torch.arange(V, device=device).unsqueeze(0).expand(bs, -1)
        scores = agent.rank(seqs_t, all_items)                    # (B, V)

        # Correctness guard: NaN/inf in scores silently poisons rank=1 via
        # (NaN > NaN) == False short-circuits, fabricating HR@1. Crash loudly.
        if not torch.isfinite(scores).all():
            n_bad = int((~torch.isfinite(scores)).any(dim=1).sum().item())
            raise RuntimeError(
                f"Non-finite ranking scores for {n_bad}/{scores.size(0)} users "
                "- likely encoder NaN from padding masking. Check SASRec attention path.")

        # mask seen items per user
        for i, seen in enumerate(seen_sets):
            if seen:
                scores[i, list(seen)] = -float("inf")

        # extract rank of true + top-K items
        true_t = torch.as_tensor(t_batch, device=device, dtype=torch.long).unsqueeze(1)
        true_score = scores.gather(1, true_t)                     # (B, 1)
        if not torch.isfinite(true_score).all():
            raise RuntimeError("Non-finite score on true test item - data pipeline bug.")
        # rank of the true = 1 + #items strictly scored higher
        rnk = (scores > true_score).sum(dim=1) + 1
        topk_idx = scores.topk(max_k, dim=1).indices             # (B, max_k)

        ranks.extend(rnk.cpu().numpy().tolist())
        topk_np = topk_idx.cpu().numpy()
        topk_items_per_user.extend(list(topk_np))
        user_is_tail.extend([int(item_pop[int(t)] <= tail_cutoff) for t in t_batch])
        coverage_set.update(topk_np.reshape(-1).tolist())

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    tail_mask = np.asarray(user_is_tail, dtype=bool)

    out: Dict[str, float] = {}
    for k in topks:
        hit = (ranks_arr <= k).astype(np.float64)
        ndcg = np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1.0), 0.0)
        out[f"HR@{k}"]   = float(hit.mean())
        out[f"NDCG@{k}"] = float(ndcg.mean())
        if tail_mask.any():
            out[f"TailHR@{k}"] = float(hit[tail_mask].mean())
            out[f"TailNDCG@{k}"] = float(ndcg[tail_mask].mean())
    out["MRR"] = float((1.0 / ranks_arr).mean())
    out["MeanRank"] = float(ranks_arr.mean())
    out["Coverage"] = float(len(coverage_set) / max(V - 1, 1))

    # novelty & diversity at the largest K
    if len(topk_items_per_user) > 0:
        K = max_k
        nov = []
        div = []
        # load item embeddings if available (agent.encoder.item_emb)
        emb_w = None
        if hasattr(agent, "encoder") and hasattr(agent.encoder, "item_emb"):
            emb_w = agent.encoder.item_emb.weight.detach()
        elif hasattr(agent, "online") and hasattr(agent.online, "item_emb"):
            emb_w = agent.online.item_emb.weight.detach()

        for lst in topk_items_per_user:
            nov.append(float(np.mean(-np.log(pop_prob[lst] + 1e-12))))
            if emb_w is not None and len(lst) > 1:
                e = emb_w[torch.as_tensor(lst, device=emb_w.device)]
                e = torch.nn.functional.normalize(e, dim=-1)
                sim = (e @ e.T)
                n = sim.size(0)
                off = (sim.sum() - n) / (n * (n - 1))
                div.append(float(1.0 - off.item()))
        if nov: out[f"Novelty@{K}"] = float(np.mean(nov))
        if div: out[f"Diversity@{K}"] = float(np.mean(div))

    return out


@torch.no_grad()
def coverage_at_alpha(r_sim: np.ndarray, r_real: np.ndarray,
                      q_hat: float) -> float:
    """Empirical P(|r_sim - r_real| <= q_hat). Should be >= 1-alpha if CRC is valid."""
    return float((np.abs(r_sim - r_real) <= q_hat).mean())
