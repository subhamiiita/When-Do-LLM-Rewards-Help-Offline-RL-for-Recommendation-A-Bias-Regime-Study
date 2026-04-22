"""SASRec (Kang & McAuley, 2018) for the f_sta statistical base model.

Minimal faithful reimplementation:
 * Item embedding table (n_items + 1 for pad id = 0).
 * Learnable positional embedding of length max_len.
 * N transformer blocks (causal self-attention).
 * Prediction head = dot product between final-position hidden and item embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SASRecConfig:
    n_items: int
    max_len: int = 50
    d_model: int = 64
    n_heads: int = 2
    n_blocks: int = 2
    dropout: float = 0.2


class SASRecBlock(nn.Module):
    def __init__(self, d: int, h: int, p: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.Dropout(p))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class SASRec(nn.Module):
    """Note: item ids are 0..n_items-1 from dataset; we reserve a PAD id at index n_items."""

    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_id = cfg.n_items  # extra slot at the end
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([SASRecBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_blocks)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _masks(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = seq.size(1)
        attn_mask = torch.triu(torch.ones(T, T, device=seq.device, dtype=torch.bool), diagonal=1)
        key_padding = seq == self.pad_id
        return attn_mask, key_padding

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T) item ids (use self.pad_id for padding). Returns (B, T, d).

        We zero out pad-position outputs after each block so NaNs from all-masked
        attention softmax (which happen at pad queries attending only to pad keys)
        cannot propagate into later blocks via the residual stream.
        """
        B, T = seq.shape
        pos = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)
        attn_mask, kpm = self._masks(seq)
        non_pad = (~kpm).unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        x = x * non_pad
        for blk in self.blocks:
            x = blk(x, attn_mask, kpm)
            x = torch.nan_to_num(x, nan=0.0) * non_pad
        return self.ln(x)

    def last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.encode(seq)  # (B, T, d)
        # Index of last non-pad position per row
        non_pad = seq != self.pad_id
        lengths = non_pad.sum(dim=1).clamp(min=1) - 1  # (B,)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    def score_items(self, hidden: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """hidden: (B, d); item_ids: (B, K) → (B, K) dot-product logits."""
        emb = self.item_emb(item_ids)  # (B, K, d)
        return torch.einsum("bd,bkd->bk", hidden, emb)

    def score_all(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, d) → (B, n_items) over full catalog."""
        w = self.item_emb.weight[: self.cfg.n_items]  # exclude pad row
        return hidden @ w.t()

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward: predict pos vs neg at every step.
        seq, pos, neg all (B, T). Returns pos_logits, neg_logits, mask (where seq != pad).
        """
        h = self.encode(seq)  # (B, T, d)
        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)
        pos_logits = (h * pos_emb).sum(dim=-1)
        neg_logits = (h * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits
