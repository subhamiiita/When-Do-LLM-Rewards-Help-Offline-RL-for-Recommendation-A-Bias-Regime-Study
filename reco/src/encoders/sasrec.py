"""SASRec (Kang & McAuley, ICDM 2018) — shared sequence encoder.

Outputs per-position hidden states h_t that encode "state" for RL and
serve as query vectors for next-item scoring in the supervised warmup.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.c1 = nn.Conv1d(dim, dim, 1)
        self.c2 = nn.Conv1d(dim, dim, 1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = x.transpose(1, 2)
        h = self.drop1(self.act(self.c1(h)))
        h = self.drop2(self.c2(h))
        return h.transpose(1, 2) + x


class SASRecBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = PointWiseFeedForward(dim, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor,
                key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        # At padded query positions the causal+key_padding masks jointly cover
        # zero valid keys -> softmax([-inf,...]) = NaN. Zero out those rows so
        # NaN never enters the residual stream and contaminates later blocks.
        if key_padding_mask is not None:
            a = a.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + a
        x = self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    """Causal self-attention encoder over item sequences.

    forward(seq, pad_mask) -> hidden states (B, L, D)
    score(seq, candidates) -> logits (B, L, K) for candidate items
    """

    def __init__(self, num_items: int, hidden_dim: int = 64,
                 max_seq_len: int = 50, num_blocks: int = 2, num_heads: int = 2,
                 dropout: float = 0.2, attn_dropout: float = 0.2,
                 item_emb_init: torch.Tensor | None = None):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_ln = nn.LayerNorm(hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_blocks)
        ])
        self.out_ln = nn.LayerNorm(hidden_dim)

        self._init_weights(item_emb_init=item_emb_init)

    def _init_weights(self, item_emb_init: torch.Tensor | None = None):
        if item_emb_init is not None:
            if item_emb_init.shape != self.item_emb.weight.shape:
                raise ValueError(
                    f"item_emb_init shape {tuple(item_emb_init.shape)} does not "
                    f"match item_emb.weight {tuple(self.item_emb.weight.shape)}")
            with torch.no_grad():
                self.item_emb.weight.data.copy_(item_emb_init)
        else:
            nn.init.trunc_normal_(self.item_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _build_masks(self, seq: torch.Tensor):
        B, L = seq.shape
        device = seq.device
        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        pad = (seq == 0)  # (B, L) True where pad
        return causal, pad

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_ln(x))

        causal, pad = self._build_masks(seq)
        for blk in self.blocks:
            x = blk(x, attn_mask=causal, key_padding_mask=pad)
        return self.out_ln(x)  # (B, L, D)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode(seq)

    def score_all(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits over the full item table. hidden: (B, L, D) -> (B, L, V)."""
        return hidden @ self.item_emb.weight.T

    def score_candidates(self, hidden: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, D), cand: (B, K) -> (B, L, K)."""
        e = self.item_emb(cand)  # (B, K, D)
        return torch.einsum("bld,bkd->blk", hidden, e)

    def last_state(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pick the hidden state at the last valid position. lengths: (B,) in [1..L]."""
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)  # (B, D)
