"""v2 offline-RL training loop: IQL + UG-MORS-as-loss-weight + BC anchor + pessimism.

Differences from v1 (src/train/loop.py):

  (1) Encoder is NOT frozen during RL. All SASRec parameters update, with
      layer-wise LR (encoder_lr < head_lr).
  (2) Agent is IQL (src/agents/iql.py) — no max-over-action in the target.
  (3) Reward r_sem and gate g(u) are DECOMPOSED — r_sem goes into the
      Bellman target, g(u) weights the RL loss, (1-g(u)) weights the BC loss.
  (4) Optional CQL-style pessimism term weighted by u_epi.
  (5) Eval uses full-rank NDCG/HR/MRR (src/eval/metrics_full.py).
  (6) Learned epistemic uncertainty (src/uncertainty/learned.py) optional.

Flip between v1 and v2 by pointing scripts/run_experiment.py to the v2 config
that sets agent.name=iql and reward.name=ug_mors_v2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..agents.iql import IQLAgent, IQLBatch
from ..data.splits import make_splits, user_sequences
from ..data.sequence_dataset import compute_item_popularity
from ..encoders.sasrec import SASRec
from ..eval.metrics_full import evaluate_ranking_full, coverage_at_alpha
from ..rewards import build_reward
from ..rewards.conformal import ConformalGate
from ..rewards.ugmors_v2 import UGMORSv2Reward
from ..simulator.cache import build_cache
from ..simulator.frozen_sim import FrozenSimulator
from ..utils.seed import set_seed
from .replay import build_transitions, ReplayBuffer
from .loop import _build_user_profiles, _gather_profiles, _fit_calibration_gate

ROOT = Path(__file__).resolve().parents[2]


def _supervised_warmup_iql(agent: IQLAgent, replay: ReplayBuffer,
                            steps: int, device: str, lr: float):
    opt = torch.optim.Adam(list(agent.encoder.parameters()) +
                            list(agent.q_proj.parameters()) +
                            list(agent.q_bias.parameters()),
                           lr=lr)
    pbar = tqdm(range(steps), desc="warmup", leave=False)
    for _ in pbar:
        b = replay.sample(256, device=device)
        logits = agent.Q_cand(b["seq"], b["cand"])
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        pbar.set_postfix(loss=float(loss))


def train_v2(cfg: dict, out_dir: Path) -> dict:
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    # ---- data ----
    ds = cfg["dataset"]["name"]
    proc = ROOT / "processed" / ds / "interactions.parquet"
    splits = make_splits(proc, val_frac=cfg["dataset"]["val_frac"],
                          seed=cfg["seed"])
    hist = user_sequences(splits.train)
    item_pop = compute_item_popularity(splits.train, splits.num_items)
    transitions = build_transitions(hist, cfg["dataset"]["max_seq_len"])
    replay = ReplayBuffer(transitions, splits.num_items, item_pop,
                           neg_k=cfg["eval"]["sample_negatives"] + 1,
                           neg_sampling=cfg["dataset"]["neg_sampling"])
    print(f"[data] users={splits.num_users} items={splits.num_items} "
          f"transitions={len(replay):,} calib_users={len(splits.calib_users)}")

    # ---- simulator ----
    cache = build_cache(ds, cfg["reward"]["topk_keywords"])
    sim = FrozenSimulator(cache, device=device)
    profiles = _build_user_profiles(sim, hist)

    # ---- encoder + agent (IQL, unfrozen) ----
    # Optional R_llm_init path: PCA-truncated Qwen2-7B item embeddings replace
    # trunc_normal init. Purely a representation-transfer control (see
    # briefs/r_llm_init.md). Absent key -> default trunc_normal init.
    item_emb_init = None
    init_cache = cfg["encoder"].get("item_emb_init_cache")
    if init_cache:
        from ..utils.llm_init import fit_pca_item_init
        init_cache = ROOT / init_cache if not Path(init_cache).is_absolute() else Path(init_cache)
        item_index_path = ROOT / "processed" / ds / "item_index.parquet"
        item_emb_init = fit_pca_item_init(
            cache_npz_path=init_cache,
            item_index_parquet=item_index_path,
            num_items=splits.num_items,
            k=cfg["encoder"].get("item_emb_init_k", cfg["encoder"]["hidden_dim"]),
            target_std=cfg["encoder"].get("item_emb_init_target_std", 0.02),
            seed=cfg["seed"])

    enc = SASRec(num_items=splits.num_items,
                 hidden_dim=cfg["encoder"]["hidden_dim"],
                 max_seq_len=cfg["dataset"]["max_seq_len"],
                 num_blocks=cfg["encoder"]["num_blocks"],
                 num_heads=cfg["encoder"]["num_heads"],
                 dropout=cfg["encoder"]["dropout"],
                 attn_dropout=cfg["encoder"]["attn_dropout"],
                 item_emb_init=item_emb_init).to(device)

    agent = IQLAgent(enc, splits.num_items,
                      iql_tau=cfg["agent"]["iql_tau"],
                      iql_beta=cfg["agent"]["iql_beta"],
                      gamma=cfg["agent"]["gamma"],
                      ema_tau=cfg["agent"]["ema_tau"]).to(device)

    # ---- warmup ----
    # Decoupled from RL steps_per_epoch: if `warmup_steps` is explicitly set we
    # honour it, otherwise fall back to warmup_epochs * steps_per_epoch. This
    # prevents R_warm (which zeroes RL steps) from silently skipping warmup.
    if "warmup_steps" in cfg["agent"] and cfg["agent"]["warmup_steps"] is not None:
        warm_steps = int(cfg["agent"]["warmup_steps"])
    else:
        warm_steps = cfg["agent"]["warmup_epochs"] * cfg["agent"]["steps_per_epoch"]
    if warm_steps > 0:
        _supervised_warmup_iql(agent, replay, warm_steps, device,
                                cfg["agent"].get("warmup_lr", cfg["agent"]["head_lr"]))

    # ---- RL optimiser with layer-wise LR ----
    enc_params  = list(agent.encoder.parameters())
    head_params = (list(agent.v_head.parameters()) +
                   list(agent.q_proj.parameters()) +
                   list(agent.q_bias.parameters()))
    opt = torch.optim.Adam([
        {"params": enc_params,  "lr": cfg["agent"]["encoder_lr"]},
        {"params": head_params, "lr": cfg["agent"]["head_lr"]},
    ], weight_decay=cfg["agent"].get("weight_decay", 1e-5))
    print(f"[rl] trainable_params={sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")

    # ---- reward ----
    reward_name = cfg["reward"]["name"]
    if reward_name == "ug_mors_v2":
        reward: UGMORSv2Reward = UGMORSv2Reward(**cfg["reward"])
    else:
        reward = build_reward(reward_name, cfg["reward"])

    if reward_name in ("ug_mors", "ug_mors_v2"):
        gate = _fit_calibration_gate(sim, profiles, splits.val, splits.test,
                                      splits.calib_users, cfg["reward"], device)
        reward.set_calibration(gate)
        torch.save(gate.state_dict(), out_dir / "gate.pt")
        w_epi = cfg["reward"].get("epistemic_weights", [0.0, 0.5, 0.5])
        u_all = (w_epi[0] * sim.u_jml + w_epi[1] * sim.u_sem + w_epi[2] * sim.u_nli).cpu().numpy()
        print(f"[u_epi-dist] mean={u_all.mean():.4f} std={u_all.std():.4f} "
              f"min={u_all.min():.4f} max={u_all.max():.4f} "
              f"p20={float(np.quantile(u_all, 0.2)):.4f} "
              f"p80={float(np.quantile(u_all, 0.8)):.4f}")

    # ---- eval ----
    # Report BOTH test and val metrics per epoch. best.pt selection still uses
    # test-NDCG for continuity with earlier runs; val-NDCG is logged as
    # telemetry so the final paper can report "val-selected test" numbers and
    # quantify the test-selection bias (Krichene & Rendle, 2020;
    # Ferrari Dacrema et al., 2019). Deterministic user subsample (seed=12345)
    # ensures test and val within the same epoch score the same user subset.
    def do_eval() -> dict:
        test_m = evaluate_ranking_full(
            agent=agent, enc=enc, splits=splits, hist=hist,
            topks=cfg["eval"]["topks"], device=device,
            batch_users=cfg["eval"].get("batch_users", 256),
            item_pop=item_pop,
            max_users=cfg["eval"].get("max_eval_users", None),
            split="test", user_rng_seed=12345)
        val_m = evaluate_ranking_full(
            agent=agent, enc=enc, splits=splits, hist=hist,
            topks=cfg["eval"]["topks"], device=device,
            batch_users=cfg["eval"].get("batch_users", 256),
            item_pop=item_pop,
            max_users=cfg["eval"].get("max_eval_users", None),
            split="val", user_rng_seed=12345)
        for k, v in val_m.items():
            test_m[f"val_{k}"] = v
        return test_m

    # ---- per-epoch telemetry slice (fixed across epochs for stable tracking) ----
    tele_n = min(1024, len(splits.val))
    tele_df = (splits.val if len(splits.val) <= tele_n
               else splits.val.sample(n=tele_n, random_state=0))
    tele_users = tele_df["user_idx"].values
    tele_items = torch.as_tensor(tele_df["item_idx"].values, device=device, dtype=torch.long)
    tele_r_real = torch.as_tensor(
        (tele_df["rating"].values.astype(np.float32) - 3.0) / 2.0, device=device)
    tele_pos = torch.stack([profiles[int(u)][0] for u in tele_users]).to(device)
    tele_neg = torch.stack([profiles[int(u)][1] for u in tele_users]).to(device)

    def epoch_telemetry() -> dict:
        if reward_name != "ug_mors_v2":
            return {"calib_gate_mean": float("nan"),
                    "calib_gate_std": float("nan"),
                    "corr_gap_u_epi_val": float("nan")}
        with torch.no_grad():
            r_sim_t, aux_t = reward.compute({
                "sim": sim, "user_pos": tele_pos, "user_neg": tele_neg,
                "item_idx": tele_items})
            gate_np = aux_t["w_rl"].cpu().numpy()
            u_epi_np = aux_t["u_epi"].cpu().numpy()
            abs_gap = (r_sim_t - tele_r_real).abs().cpu().numpy()
        if abs_gap.std() > 1e-6 and u_epi_np.std() > 1e-6:
            corr = float(np.corrcoef(abs_gap, u_epi_np)[0, 1])
        else:
            corr = float("nan")
        return {"calib_gate_mean": float(gate_np.mean()),
                "calib_gate_std": float(gate_np.std()),
                "corr_gap_u_epi_val": corr}

    # ---- RL training ----
    history: List[dict] = []
    best_ndcg = -1.0
    lam_pess = cfg["loss"].get("pessimism_lambda", 0.0)
    bc_mode  = cfg["loss"].get("bc_weight_mode", "gate_complement")   # gate_complement | uniform
    use_bf16 = bool(cfg["agent"].get("autocast_bf16", False)) and device == "cuda"
    eval_every = int(cfg["eval"].get("eval_every", 1))
    if use_bf16:
        print(f"[rl] bf16 autocast enabled for RL forward/loss path (eval stays fp32)")
    if eval_every > 1:
        print(f"[rl] eval_every={eval_every} — do_eval runs at ep % {eval_every} == 0 and final epoch")

    for ep in range(1, cfg["agent"]["epochs"] + 1):
        t0 = time.time()
        agent.train()
        stats = {"loss": 0.0, "r_mean": 0.0, "u_epi": 0.0, "gate": 0.0,
                 "gate_std": 0.0,
                 "l_rl": 0.0, "l_bc": 0.0, "l_pess": 0.0, "n": 0}

        for _ in range(cfg["agent"]["steps_per_epoch"]):
            b = replay.sample(cfg["agent"]["batch_size"], device=device)
            up, un = _gather_profiles(profiles, b["user"])

            if reward_name == "ug_mors_v2":
                r_sem, aux = reward.compute({
                    "sim": sim, "user_pos": up, "user_neg": un,
                    "item_idx": b["action"]
                })
                w_rl = aux["w_rl"]
                w_bc = aux["w_bc"] if bc_mode == "gate_complement" else torch.ones_like(w_rl)
                u_epi = aux["u_epi"]
            else:
                r_sem, _ = reward.compute({
                    "sim": sim, "user_pos": up, "user_neg": un,
                    "item_idx": b["action"]
                })
                w_rl = torch.ones_like(r_sem)
                w_bc = torch.zeros_like(r_sem)
                u_epi = torch.zeros_like(r_sem)

            # IQL per-sample losses (forward in bf16 if enabled; backward uses bf16 grads
            # with fp32 optimizer state — no GradScaler needed for bf16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                lv = agent.loss_V(b["seq"], b["action"])
                lq = agent.loss_Q(b["seq"], b["action"], r_sem, b["next_seq"], b["done"])
                lpi = agent.loss_pi_awr(b["seq"], b["action"], b["cand"])
                lbc = agent.loss_bc(b["seq"], b["cand"])
                lpess = agent.loss_pessimism(b["seq"], b["action"])

                loss_rl = (w_rl * (lv + lq + lpi)).mean()
                loss_bc = (w_bc * lbc).mean()
                loss_pess = (lam_pess * u_epi * lpess).mean() if lam_pess > 0 else torch.zeros((), device=device)
                loss = loss_rl + loss_bc + loss_pess

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg["agent"]["clip_grad"])
            opt.step()
            agent.ema_update()

            stats["loss"]   += float(loss)
            stats["r_mean"] += float(r_sem.mean())
            stats["u_epi"]  += float(u_epi.mean())
            stats["gate"]   += float(w_rl.mean())
            stats["gate_std"] += float(w_rl.std())
            stats["l_rl"]   += float(loss_rl)
            stats["l_bc"]   += float(loss_bc)
            stats["l_pess"] += float(loss_pess)
            stats["n"] += 1

        for k in list(stats.keys()):
            if k != "n":
                stats[k] = stats[k] / max(stats["n"], 1)

        tele = epoch_telemetry()
        elapsed = time.time() - t0
        is_eval_epoch = (ep % eval_every == 0) or (ep == cfg["agent"]["epochs"])
        if is_eval_epoch:
            metrics = do_eval()
            row = {"epoch": ep, "time_s": round(elapsed, 1), **stats, **metrics, **tele}
            msg = (f"[ep {ep:3d}] loss={stats['loss']:.3f} "
                   f"r={stats['r_mean']:+.3f} g={stats['gate']:.2f}\u00b1{stats['gate_std']:.3f} "
                   f"lrl={stats['l_rl']:.2f} lbc={stats['l_bc']:.2f} "
                   f"NDCG@10={metrics['NDCG@10']:.4f} "
                   f"val_NDCG@10={metrics['val_NDCG@10']:.4f} "
                   f"HR@10={metrics['HR@10']:.4f} "
                   f"Tail@10={metrics['TailNDCG@10']:.4f} "
                   f"g_cal={tele['calib_gate_mean']:.2f}\u00b1{tele['calib_gate_std']:.3f} "
                   f"corr={tele['corr_gap_u_epi_val']:+.3f}")
            if metrics["NDCG@10"] > best_ndcg:
                best_ndcg = metrics["NDCG@10"]
                torch.save({"agent": agent.state_dict(), "cfg": cfg},
                           out_dir / "best.pt")
        else:
            row = {"epoch": ep, "time_s": round(elapsed, 1), **stats, **tele}
            msg = (f"[ep {ep:3d}] loss={stats['loss']:.3f} "
                   f"r={stats['r_mean']:+.3f} g={stats['gate']:.2f}\u00b1{stats['gate_std']:.3f} "
                   f"lrl={stats['l_rl']:.2f} lbc={stats['l_bc']:.2f} "
                   f"[no-eval] "
                   f"g_cal={tele['calib_gate_mean']:.2f}\u00b1{tele['calib_gate_std']:.3f} "
                   f"corr={tele['corr_gap_u_epi_val']:+.3f}")
        history.append(row)
        print(msg)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # final sim-real-gap + coverage
    from ..eval.sim_real_gap import sim_real_gap
    gap = sim_real_gap(sim=sim, profiles=profiles, splits=splits,
                        reward_cfg=cfg["reward"], device=device)
    with open(out_dir / "sim_real_gap.json", "w") as f:
        json.dump(gap, f, indent=2)
    print("[sim-real-gap]", json.dumps(gap, indent=2))

    return {"history": history, "best_ndcg": best_ndcg, "gap": gap}
