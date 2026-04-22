"""Train q_leave LSTM: label = 1 if current step is within last-k of user's full seq, else 0."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.common.config import dataset_config, project_root
from src.common.device import get_device
from src.common.seed import set_seed
from src.data.loaders import load_splits
from src.models.retention import RetentionConfig, RetentionLSTM


class RetentionDataset(Dataset):
    """Produces many (prefix, label) pairs: at each step in a user's sequence, the
    prefix is items up to that step, label = 1 if step is within last k of the full
    sequence length.
    """

    def __init__(self, sequences, n_items: int, max_len: int, k_last: int):
        self.examples = []
        self.pad = n_items
        for seq in sequences:
            L = len(seq)
            if L < 2:
                continue
            iids = [s[0] for s in seq]
            for t in range(1, L):  # predict after observing items [:t]
                label = 1 if (L - t) <= k_last else 0
                self.examples.append((iids[:t], label))
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        prefix, label = self.examples[idx]
        seq = np.full(self.max_len, self.pad, dtype=np.int64)
        take = prefix[-self.max_len :]
        seq[-len(take):] = take
        return torch.from_numpy(seq), float(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = dataset_config(args.dataset)
    device = get_device()
    splits = load_splits(args.dataset)

    rc = cfg["retention"]
    max_len = cfg["sasrec"]["max_len"]

    ds = RetentionDataset(splits["sequences"], splits["n_items"], max_len, rc["k_last"])
    n = len(ds)
    n_val = max(1, int(n * 0.05))
    perm = torch.randperm(n)
    val_idx = set(perm[:n_val].tolist())
    train_examples = [ex for i, ex in enumerate(ds.examples) if i not in val_idx]
    val_examples = [ex for i, ex in enumerate(ds.examples) if i in val_idx]

    class _Sub(Dataset):
        def __init__(self, exs):
            self.exs = exs

        def __len__(self):
            return len(self.exs)

        def __getitem__(self, i):
            prefix, label = self.exs[i]
            seq = np.full(max_len, splits["n_items"], dtype=np.int64)
            take = prefix[-max_len:]
            seq[-len(take):] = take
            return torch.from_numpy(seq), float(label)

    tr_dl = DataLoader(_Sub(train_examples), batch_size=rc["batch_size"], shuffle=True, num_workers=0, drop_last=False)
    va_dl = DataLoader(_Sub(val_examples), batch_size=rc["batch_size"] * 4, shuffle=False)

    model = RetentionLSTM(RetentionConfig(n_items=splits["n_items"], max_len=max_len, d_model=rc["hidden"], hidden=rc["hidden"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rc["lr"])
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()

    ckpt_path = project_root() / cfg["paths"]["retention_ckpt"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    for epoch in range(rc["epochs"]):
        model.train()
        tot, steps = 0.0, 0
        for seq, lab in tr_dl:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model.logits(seq)
                loss = bce(lg.float(), lab.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item())
            steps += 1

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for seq, lab in va_dl:
                seq = seq.to(device, non_blocking=True)
                p = model(seq).float().cpu().numpy()
                ps.append(p)
                ys.append(lab.numpy())
        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        try:
            auc = float(roc_auc_score(ys, ps))
        except Exception:
            auc = float("nan")
        dt = time.time() - t0
        print(f"[retention {args.dataset}] epoch {epoch+1:02d}/{rc['epochs']} loss={tot/max(1,steps):.4f} val_auc={auc:.4f} ({dt:.0f}s)")
        if auc > best_auc:
            best_auc = auc
            torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__, "val_auc": auc}, ckpt_path)
    print(f"[retention {args.dataset}] best AUC={best_auc:.4f} saved to {ckpt_path}")


if __name__ == "__main__":
    main()
