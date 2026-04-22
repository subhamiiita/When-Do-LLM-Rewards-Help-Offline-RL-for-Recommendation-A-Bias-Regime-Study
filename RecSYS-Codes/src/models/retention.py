"""q_leave LSTM: predict probability the current step is within the last-k interactions
of the user's full sequence. Used as r_ret proxy in UG-MORS."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RetentionConfig:
    n_items: int
    max_len: int = 50
    d_model: int = 64
    hidden: int = 64


class RetentionLSTM(nn.Module):
    def __init__(self, cfg: RetentionConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.n_items
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d_model, padding_idx=self.pad_id)
        self.lstm = nn.LSTM(cfg.d_model, cfg.hidden, batch_first=True)
        self.head = nn.Linear(cfg.hidden, 1)

    def logits(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T). Returns (B,) pre-sigmoid logits on last non-pad position."""
        x = self.item_emb(seq)
        out, _ = self.lstm(x)
        lengths = (seq != self.pad_id).sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(1, idx).squeeze(1)
        return self.head(last).squeeze(-1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(seq))
