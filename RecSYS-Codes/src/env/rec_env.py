"""Vectorized recsys RL environment.

 * Torch-native (no gym) to avoid Python overhead.
 * Batch of N parallel trajectories, one user each.
 * State representation returned to the agent = frozen SASRec last_hidden (B, d_model).
 * Reward plugged in via `src.reward.dispatch.compute_reward(variant, ...)`.
 * Pseudo-label for history append = baseline vote (consistent across variants so the
   history evolves identically regardless of reward choice).
 * Episode terminates after `horizon` steps or when the environment resets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.reward.baseline_vote import compute_baseline_vote
from src.reward.common import RewardEngine, RewardState
from src.reward.dispatch import compute_reward
from src.reward.margins import compute_margins


@dataclass
class EnvConfig:
    n_envs: int = 32
    horizon: int = 20
    seed: int = 0
    min_history: int = 5          # minimum history length to sample a user from
    div_k: int = 3


class RecEnv:
    """Batched env. step() takes an (n_envs,) action tensor and returns (reward, done)."""

    def __init__(self, engine: RewardEngine, variant: str, cfg: EnvConfig):
        self.engine = engine
        self.variant = variant
        self.cfg = cfg
        self.device = engine.device

        self.splits = engine.splits
        self.n_users = self.splits["n_users"]
        self.n_items = self.splits["n_items"]
        self.max_len = engine.sas_max_len
        self.pad = engine.sas_pad

        # --- per-env buffers (persist across steps) -----------------------
        N = cfg.n_envs
        self.hist_ids = torch.full((N, self.max_len), self.pad, dtype=torch.long, device=self.device)
        self.hist_labels = torch.zeros((N, self.max_len), dtype=torch.long, device=self.device)
        self.hist_mask = torch.zeros((N, self.max_len), dtype=torch.bool, device=self.device)
        self.user_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self.t = torch.zeros(N, dtype=torch.long, device=self.device)
        self.recent_actions = torch.full((N, cfg.div_k), -1, dtype=torch.long, device=self.device)

        # --- user eligibility pool --------------------------------------
        self._eligible_users = [
            u for u in range(self.n_users) if len(self.splits["train"][u]) >= cfg.min_history
        ]
        if not self._eligible_users:
            raise ValueError("No users with enough history; lower min_history or check dataset.")
        self.rng = np.random.default_rng(cfg.seed)

        # Cache all user histories as arrays (speeds up reset)
        self._user_seq_arrays: dict[int, np.ndarray] = {}
        for u in self._eligible_users:
            seq = self.splits["train"][u]
            iids = np.array([s[0] for s in seq], dtype=np.int64)
            labs = np.array([s[1] for s in seq], dtype=np.int64)
            self._user_seq_arrays[u] = np.stack([iids, labs], axis=0)  # (2, L)

        # --- precompute margins helper needs nothing stateful
        self.reset()

    # ---------------------------------------------------------------------
    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset specified envs with a random user & a random starting history length."""
        for i in env_ids.tolist():
            u = int(self.rng.choice(self._eligible_users))
            arr = self._user_seq_arrays[u]
            L = arr.shape[1]
            # Random starting point: keep the first ~min_history..L-1 items as history, roll from there
            start = int(self.rng.integers(self.cfg.min_history, max(L, self.cfg.min_history + 1)))
            take = min(start, self.max_len)
            window_iids = arr[0, start - take : start]
            window_labs = arr[1, start - take : start]

            self.hist_ids[i].fill_(self.pad)
            self.hist_labels[i].fill_(0)
            self.hist_mask[i].fill_(False)
            if take > 0:
                self.hist_ids[i, self.max_len - take :] = torch.from_numpy(window_iids).to(self.device)
                self.hist_labels[i, self.max_len - take :] = torch.from_numpy(window_labs).to(self.device)
                self.hist_mask[i, self.max_len - take :] = True
            self.user_idx[i] = u
            self.t[i] = 0
            self.recent_actions[i].fill_(-1)

    def reset(self) -> None:
        """Full reset of all envs."""
        self._reset_envs(torch.arange(self.cfg.n_envs, device=self.device))

    # ---------------------------------------------------------------------
    def _state(self) -> RewardState:
        return RewardState(
            hist_ids=self.hist_ids,
            hist_labels=self.hist_labels,
            hist_mask=self.hist_mask,
            user_idx=self.user_idx,
            t=self.t,
            recent_actions=self.recent_actions,
        )

    def state_repr(self) -> torch.Tensor:
        """Frozen SASRec encoding as state features for the agent. Returns (N, d_model)."""
        with torch.no_grad():
            return self.engine.sasrec.last_hidden(self.hist_ids)

    def valid_action_mask(self) -> torch.Tensor:
        """Optional: mask items already seen in history so agent doesn't recommend duplicates.
        Returns (N, n_items) bool; True = valid."""
        N = self.cfg.n_envs
        mask = torch.ones((N, self.n_items), dtype=torch.bool, device=self.device)
        # zero-out items in current history
        # index_put with (env_ids, item_ids)
        rows = torch.arange(N, device=self.device).unsqueeze(1).expand_as(self.hist_ids)
        valid_hist = self.hist_mask
        rows_flat = rows[valid_hist]
        items_flat = self.hist_ids[valid_hist]
        mask[rows_flat, items_flat] = False
        return mask

    # ---------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """action: (N,) long item ids.
        Returns (reward, done, next_state_repr).
        """
        N = self.cfg.n_envs
        assert action.shape == (N,)

        state_prev = self._state()

        # --- reward ---------------------------------------------------
        # For ug_pbrs we need both prev and next state. Compute reward AFTER history update.
        # So: compute everything except pbrs now, and for pbrs branch after the update.

        # Baseline vote is always computed because it supplies the pseudo-label for history append.
        margins = compute_margins(self.engine, state_prev, action)
        bv = compute_baseline_vote(self.engine, state_prev, action, margins=margins)
        pseudo_label = bv["vote"]                                   # (N,) 0/1

        if self.variant == "ug_pbrs":
            # Need next state → update history first, compute Phi on both
            saved = self._clone_state()
            self._append_to_history(action, pseudo_label)
            state_next = self._state()
            reward_dict = _lazy_ug_pbrs(self.engine, state_prev_snapshot=saved, state_next=state_next, action=action, gamma=0.9)
            reward = reward_dict["R"]
        else:
            reward = compute_reward(self.variant, self.engine, state_prev, action, prev_state=None)
            self._append_to_history(action, pseudo_label)

        # Update recent actions ring buffer
        self.recent_actions = torch.roll(self.recent_actions, shifts=1, dims=1)
        self.recent_actions[:, 0] = action
        self.t = self.t + 1

        # Done flag: episodes end at horizon
        done = self.t >= self.cfg.horizon
        if done.any():
            env_ids = torch.nonzero(done, as_tuple=False).flatten()
            self._reset_envs(env_ids)

        # Next-state representation
        next_repr = self.state_repr()
        return reward.float(), done, next_repr

    def _append_to_history(self, action: torch.Tensor, label: torch.Tensor) -> None:
        """Shift history left by 1 and append (action, label) at the right end."""
        self.hist_ids = torch.roll(self.hist_ids, shifts=-1, dims=1)
        self.hist_labels = torch.roll(self.hist_labels, shifts=-1, dims=1)
        self.hist_mask = torch.roll(self.hist_mask, shifts=-1, dims=1)
        self.hist_ids[:, -1] = action
        self.hist_labels[:, -1] = label
        self.hist_mask[:, -1] = True

    # ---------------------------------------------------------------------
    def _clone_state(self) -> "RewardState":
        """Snapshot current state tensors (for PBRS prev-state)."""
        return RewardState(
            hist_ids=self.hist_ids.clone(),
            hist_labels=self.hist_labels.clone(),
            hist_mask=self.hist_mask.clone(),
            user_idx=self.user_idx.clone(),
            t=self.t.clone(),
            recent_actions=self.recent_actions.clone(),
        )


def _lazy_ug_pbrs(engine, state_prev_snapshot, state_next, action, gamma):
    # Inline import to avoid circular issues at module load time.
    from src.reward.ug_pbrs import compute_ug_pbrs

    return compute_ug_pbrs(engine, state_prev_snapshot, state_next, action, gamma=gamma)
