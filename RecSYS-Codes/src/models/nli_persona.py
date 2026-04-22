"""Batched NLI scoring for (persona, item_description) pairs via DeBERTa-v3 MNLI.

Returns a SOFT score per pair in [-1, 1]:
    score = p(entailment) - p(contradiction)
Positive => persona entails item; negative => persona contradicts item; ~0 => neutral.

The paper casts r_per as {-1, 0, 1} (Eq. 8). We use the continuous logit-difference
to preserve confidence, which is consistent with UG-MORS's "dense reward" thesis and
increases r_per's SNR vs. the argmax variant.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class NLIScorer:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)
        labels = getattr(self.model.config, "id2label", {0: "entailment", 1: "neutral", 2: "contradiction"})
        self.label_map = {int(i): str(v).lower() for i, v in labels.items()}
        self.ent_idx = next(i for i, v in self.label_map.items() if v.startswith("ent"))
        self.neu_idx = next(i for i, v in self.label_map.items() if v.startswith("neu"))
        self.con_idx = next(i for i, v in self.label_map.items() if v.startswith("con"))

    @torch.no_grad()
    def score(
        self,
        premises: List[str],
        hypotheses: List[str],
        batch_size: int = 256,
        soft: bool = True,
    ) -> np.ndarray:
        """Returns float32 soft scores in [-1, 1] if soft=True, else int8 argmax in {-1, 0, 1}."""
        out: list[np.ndarray] = []
        for i in range(0, len(premises), batch_size):
            bp = premises[i : i + batch_size]
            bh = hypotheses[i : i + batch_size]
            enc = self.tok(
                bp, bh, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self.model(**enc).logits
            probs = F.softmax(logits.float(), dim=-1)  # (B, 3)
            if soft:
                s = probs[:, self.ent_idx] - probs[:, self.con_idx]  # (B,) in [-1, 1]
                out.append(s.cpu().numpy().astype(np.float32))
            else:
                pred = logits.argmax(dim=-1).cpu().numpy()
                mapped = np.where(pred == self.ent_idx, 1, np.where(pred == self.con_idx, -1, 0)).astype(np.int8)
                out.append(mapped)
        return np.concatenate(out)
