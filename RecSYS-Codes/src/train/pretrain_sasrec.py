"""Pretrain SASRec f_sta on each dataset's user sequences (all interactions, positive-only
next-item prediction with uniform negative sampling, SASRec-standard)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.common.seed import set_seed
from src.data.loaders import load_splits
from src.models.sasrec import SASRec, SASRecConfig


class SASRecSeqDataset(Dataset):
    """For each user sequence of length L, produce one training example (seq, pos, neg)
    with max_len window. Padding uses pad_id = n_items."""

    def __init__(self, train_seqs: list[list[tuple[int, int, int]]], n_items: int, max_len: int, rng_seed: int = 0):
        self.max_len = max_len
        self.pad = n_items
        self.n_items = n_items
        self.users: list[np.ndarray] = []
        for seq in train_seqs:
            if len(seq) < 2:
                continue
            iids = np.array([s[0] for s in seq], dtype=np.int64)
            self.users.append(iids)
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        iids = self.users[idx]
        L = len(iids)
        # Window last max_len + 1 items so we can build (seq, pos)
        take = iids[-(self.max_len + 1) :]
        seq = np.full(self.max_len, self.pad, dtype=np.int64)
        pos = np.full(self.max_len, self.pad, dtype=np.int64)
        n = len(take) - 1
        seq[-n:] = take[:-1]
        pos[-n:] = take[1:]
        neg = np.full(self.max_len, self.pad, dtype=np.int64)
        user_set = set(iids.tolist())
        # sample negatives where seq != pad
        for i in range(self.max_len):
            if seq[i] != self.pad:
                while True:
                    c = int(self.rng.integers(0, self.n_items))
                    if c not in user_set:
                        break
                neg[i] = c
        return torch.from_numpy(seq), torch.from_numpy(pos), torch.from_numpy(neg)


def evaluate(model: SASRec, splits: dict, device: torch.device, max_len: int, n_negs: int = 99, seed: int = 42) -> dict:
    model.eval()
    n_users = splits["n_users"]
    n_items = splits["n_items"]
    pad = model.pad_id
    val = splits["val"]
    seqs = splits["sequences"]
    rng = np.random.default_rng(seed)

    hits = 0
    ndcg_sum = 0.0
    n_eval = 0
    batch_seq, batch_cand, batch_user = [], [], []

    def flush():
        nonlocal hits, ndcg_sum, n_eval
        if not batch_seq:
            return
        seq_t = torch.stack(batch_seq).to(device)
        cand_t = torch.stack(batch_cand).to(device)
        with torch.no_grad():
            h = model.last_hidden(seq_t)
            logits = model.score_items(h, cand_t)
        order = torch.argsort(logits, dim=1, descending=True)
        pos_rank = (order == 0).float().argmax(dim=1) + 1  # positive is col 0 in candidate
        top = (pos_rank <= 10)
        hits += int(top.sum().item())
        ndcg_sum += float(torch.where(top, 1.0 / torch.log2(pos_rank.float() + 1.0), torch.zeros_like(pos_rank, dtype=torch.float32)).sum().item())
        n_eval += seq_t.size(0)
        batch_seq.clear(); batch_cand.clear(); batch_user.clear()

    for u in range(n_users):
        if val[u] is None:
            continue
        tgt = val[u][0]  # positive item idx
        hist = [it for (it, _, _) in splits["train"][u]]
        if not hist:
            continue
        seen = set(hist) | {tgt}
        cand = [tgt]
        while len(cand) < 1 + n_negs:
            c = int(rng.integers(0, n_items))
            if c not in seen:
                cand.append(c)
                seen.add(c)
        seq_arr = np.full(max_len, pad, dtype=np.int64)
        take = hist[-max_len:]
        seq_arr[-len(take):] = take
        batch_seq.append(torch.from_numpy(seq_arr))
        batch_cand.append(torch.tensor(cand, dtype=torch.long))
        batch_user.append(u)
        if len(batch_seq) >= 512:
            flush()
    flush()

    return {
        "hr@10": hits / max(1, n_eval),
        "ndcg@10": ndcg_sum / max(1, n_eval),
        "n_eval": n_eval,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = dataset_config(args.dataset)
    device = get_device()
    splits = load_splits(args.dataset)

    sa = cfg["sasrec"]
    mcfg = SASRecConfig(
        n_items=splits["n_items"],
        max_len=sa["max_len"],
        d_model=sa["d_model"],
        n_heads=sa["n_heads"],
        n_blocks=sa["n_blocks"],
        dropout=sa["dropout"],
    )
    model = SASRec(mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sa["lr"])
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    ds = SASRecSeqDataset(splits["train"], splits["n_items"], sa["max_len"], rng_seed=args.seed)
    dl = DataLoader(ds, batch_size=sa["batch_size"], shuffle=True, num_workers=0, drop_last=False)

    best_hr = -1.0
    ckpt_path = project_root() / cfg["paths"]["sasrec_ckpt"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(sa["epochs"]):
        model.train()
        tot, steps = 0.0, 0
        for seq, pos, neg in dl:
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pl, nl = model(seq, pos, neg)
                mask = (seq != model.pad_id).float()
                loss_pos = -(torch.log(torch.sigmoid(pl) + 1e-8) * mask).sum() / mask.sum().clamp(min=1)
                loss_neg = -(torch.log(1.0 - torch.sigmoid(nl) + 1e-8) * mask).sum() / mask.sum().clamp(min=1)
                loss = loss_pos + loss_neg
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
            steps += 1
        ev = evaluate(model, splits, device, max_len=sa["max_len"])
        hr = ev["hr@10"]
        dt = time.time() - t0
        print(f"[sasrec {args.dataset}] epoch {epoch+1:02d}/{sa['epochs']} loss={tot/max(1,steps):.4f} HR@10={hr:.4f} NDCG@10={ev['ndcg@10']:.4f} ({dt:.0f}s)")
        if hr > best_hr:
            best_hr = hr
            torch.save({"state_dict": model.state_dict(), "cfg": mcfg.__dict__, "val_hr10": hr}, ckpt_path)
    print(f"[sasrec {args.dataset}] best val HR@10 = {best_hr:.4f}  saved to {ckpt_path}")


if __name__ == "__main__":
    main()
