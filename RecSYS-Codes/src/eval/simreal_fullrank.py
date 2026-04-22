"""Full-rank SimReal evaluation.

Like `simreal.py` but ranks the held-out positive against the ENTIRE catalogue
(minus items already in the user's training+val history) rather than against
1 positive + 99 sampled negatives. This gives a Krichene-Rendle-consistent
NDCG/HR/MRR and is the decision signal for whether the UG-MORS lift survives
honest ranking.

Reuses the state-construction path and head-loading logic conceptually from
`simreal.py` but does not edit it. Reports NDCG@{1,5,10,20}, HR@{1,5,10,20},
and MRR as a JSON sibling of the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.data.loaders import load_splits
from src.models.sasrec import SASRec, SASRecConfig


KS = (1, 5, 10, 20)


def _dataset_from_config(config_path: Path) -> str:
    return config_path.stem


def _infer_agent(ckpt: dict) -> str:
    if "q" in ckpt:
        return "dqn"
    if "net" in ckpt:
        return "ac"
    raise ValueError(f"Unrecognised checkpoint keys: {list(ckpt.keys())}")


def _load_sasrec(dataset: str, device: torch.device) -> SASRec:
    cfg = dataset_config(dataset)
    ck = torch.load(project_root() / cfg["paths"]["sasrec_ckpt"], map_location=device, weights_only=False)
    sas = SASRec(SASRecConfig(**ck["cfg"])).to(device).eval()
    sas.load_state_dict(ck["state_dict"])
    return sas


def _build_state(sasrec: SASRec, splits: dict, users: list[int], use_val: bool, device: torch.device) -> torch.Tensor:
    """Right-aligned SASRec history tensor → last-hidden state. Mirrors simreal.py."""
    max_len = sasrec.cfg.max_len
    pad = sasrec.pad_id
    seq = torch.full((len(users), max_len), pad, dtype=torch.long, device=device)
    for i, u in enumerate(users):
        hist = [it for (it, _, _) in splits["train"][u]]
        if use_val and splits["val"][u] is not None:
            hist.append(splits["val"][u][0])
        take = hist[-max_len:]
        if take:
            seq[i, max_len - len(take):] = torch.tensor(take, dtype=torch.long, device=device)
    with torch.no_grad():
        return sasrec.last_hidden(seq)


def _load_score_fn(agent: str, ckpt_path: Path, d_state: int, n_items: int, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    from src.agents.common import HeadConfig, ItemDotActorCriticHead, ItemDotQHead

    head_cfg = HeadConfig(d_state=d_state, n_items=n_items)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if agent == "dqn":
        head = ItemDotQHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["q"])

        def score_all(s: torch.Tensor) -> torch.Tensor:
            return head(s)
    else:
        head = ItemDotActorCriticHead(head_cfg).to(device).eval()
        head.load_state_dict(ck["net"])

        def score_all(s: torch.Tensor) -> torch.Tensor:
            logits, _ = head(s)
            return logits
    return score_all


def _seen_items(splits: dict, user: int, use_val: bool) -> list[int]:
    seen = [it for (it, _, _) in splits["train"][user]]
    if use_val and splits["val"][user] is not None:
        seen.append(splits["val"][user][0])
    return seen


def evaluate_fullrank(config_path: Path, model_path: Path, out_path: Path, batch: int = 128, split: str = "test") -> dict:
    dataset = _dataset_from_config(config_path)
    device = get_device()
    splits = load_splits(dataset)
    n_items = int(splits["n_items"])
    n_users = int(splits["n_users"])
    sasrec = _load_sasrec(dataset, device)

    ck = torch.load(model_path, map_location=device, weights_only=False)
    agent = _infer_agent(ck)
    d_state = sasrec.cfg.d_model
    score_all = _load_score_fn(agent, model_path, d_state=d_state, n_items=n_items, device=device)

    target_split = splits[split]
    valid_users = [u for u in range(n_users) if target_split[u] is not None]

    use_val = (split == "test")

    ndcg_sum = {k: 0.0 for k in KS}
    hr_sum = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    n_eval = 0

    t_start = time.perf_counter()
    NEG_INF = torch.finfo(torch.float32).min

    with torch.no_grad():
        i = 0
        while i < len(valid_users):
            users = valid_users[i : i + batch]
            state = _build_state(sasrec, splits, users, use_val=use_val, device=device)
            scores = score_all(state).float()                    # (B, n_items)

            # Mask seen items (training history [+ val if split==test]) to -inf.
            for bi, u in enumerate(users):
                seen = _seen_items(splits, u, use_val=use_val)
                if seen:
                    idx = torch.tensor(seen, dtype=torch.long, device=device)
                    scores[bi, idx] = NEG_INF

            pos_ids = torch.tensor([target_split[u][0] for u in users], dtype=torch.long, device=device)
            pos_scores = scores.gather(1, pos_ids.unsqueeze(1)).squeeze(1)

            # Rank = 1 + (#items strictly ranked above the positive). Ties broken
            # adversarially (items equal to pos_score that are not the positive
            # count as "above"), which is the conservative convention.
            gt = (scores > pos_scores.unsqueeze(1)).sum(dim=1)
            eq = (scores == pos_scores.unsqueeze(1)).sum(dim=1) - 1  # subtract the positive itself
            rank = (gt + eq.clamp(min=0)).float() + 1.0              # (B,)

            mrr_sum += float((1.0 / rank).sum().item())
            for k in KS:
                in_k = rank <= k
                ndcg = torch.where(in_k, 1.0 / torch.log2(rank + 1.0), torch.zeros_like(rank))
                ndcg_sum[k] += float(ndcg.sum().item())
                hr_sum[k] += float(in_k.float().sum().item())
            n_eval += len(users)
            i += batch

    elapsed = time.perf_counter() - t_start
    denom = max(1, n_eval)
    out = dict(
        dataset=dataset,
        agent=agent,
        model_path=str(model_path),
        split=split,
        protocol="fullrank",
        n_items=n_items,
        n_eval=n_eval,
        wall_seconds=elapsed,
        ndcg_at_1=ndcg_sum[1] / denom,
        ndcg_at_5=ndcg_sum[5] / denom,
        ndcg_at_10=ndcg_sum[10] / denom,
        ndcg_at_20=ndcg_sum[20] / denom,
        hr_at_1=hr_sum[1] / denom,
        hr_at_5=hr_sum[5] / denom,
        hr_at_10=hr_sum[10] / denom,
        hr_at_20=hr_sum[20] / denom,
        mrr=mrr_sum / denom,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[fullrank {dataset}/{agent} {model_path.parent.name}] "
        f"NDCG@10={out['ndcg_at_10']:.4f} HR@10={out['hr_at_10']:.4f} "
        f"MRR={out['mrr']:.4f} n={n_eval} t={elapsed:.1f}s"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Dataset yaml, e.g. configs/ml1m.yaml")
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    evaluate_fullrank(args.config, args.model_path, args.out, batch=args.batch, split=args.split)


if __name__ == "__main__":
    main()
