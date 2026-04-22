# Uncertainty-Gated Reward Calibration for LLM-Driven Offline RL

Anonymous code release for the RecSys 2026 submission
*"When Does Uncertainty-Gated Reward Calibration Help LLM-Driven
Offline RL? A Bias-Regime Study."*

Two pipelines ship side by side:

- `reco/` — offline IQL with the soft conformal gate, SASRec encoder,
  multi-seed driver scripts, and dataset preprocessing for ML-1M,
  Yelp, and Amazon Video Games. Produces Tables 1 and 2 of the paper.
- `RecSYS-Codes/` — online-RL baselines (DQN, PPO, A2C, TRPO) over a
  frozen-SASRec simulator, sampled-99-neg and full-rank leave-last-out
  harnesses. Produces Table 3 (cross-paradigm).

Cached LLM ratings ship under `reco/cache/` so reproduction needs
no LLM API key.

## 15-minute smoke reproduction

On a single RTX 4070 (12 GB) or equivalent GPU:

```
cd reco
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements.txt
python scripts/preprocess.py --dataset ml1m
python scripts/run_primary.py --seeds 42 --datasets ml1m \
    --variants R_warm R4_ugmv2_BC
```

This trains two offline-IQL variants on ML-1M seed 42 and reports
NDCG@10 under full-rank leave-last-out. Expected rough values from
the paper (mean of three seeds): R_warm ~0.049, R4_ugmv2_BC ~0.054.
Seed-42 single-run should land within ±0.01 of the three-seed mean.

## Full reproduction

```
cd reco
bash scripts/run_all_primary.sh          # Tables 1-2, ~6 GPU-hours
cd ../RecSYS-Codes
python -m src.scripts.run_all_ablations \
    --seeds 0 1 2 --steps 50000           # Table 3, ~9 GPU-hours
python -m src.scripts.build_result_tables # aggregates CSVs
```

Precomputed `simreal_test.json` per run ships under
`RecSYS-Codes/results/runs/` so aggregation and tables can be
regenerated without retraining.

## Layout

```
reco/
  src/                 # IQL, SASRec encoder, conformal gate, eval
  configs/             # v2.yaml (main), v2_grid.yaml (sweeps)
  scripts/             # run_primary.py, run_all_primary.sh, etc.
  processed/           # preprocessed 5-core splits
  cache/               # cached LLM ratings + keyword embeddings
  requirements.txt
RecSYS-Codes/
  src/                 # DQN/PPO/A2C/TRPO agents, sim+real env, eval
  configs/             # per-dataset yaml
  scripts/             # driver scripts
  results/
    runs/              # per-run simreal_test.json
    tables/            # aggregated CSVs and markdown
```

## Environment

Python 3.12, PyTorch 2.3, CUDA 12.1. Exact pins in
`reco/requirements.txt`. Random seeds: offline IQL uses {42, 43, 44}
(ML-1M also reports {42, 7, 123} for three-seed sign tests); online
RL uses {0, 1, 2}.

## License

Code released under the MIT License on acceptance — see `LICENSE`.
Datasets retain their original licenses (ML-1M, Yelp Open Dataset
2022, Amazon Product Reviews 2018).
