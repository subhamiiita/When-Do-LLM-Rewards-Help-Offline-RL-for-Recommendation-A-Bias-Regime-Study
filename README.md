# Uncertainty-Gated Reward Calibration for LLM-Driven Offline RL

Anonymous code release for the RecSys 2026 submission
*"When Does Uncertainty-Gated Reward Calibration Help LLM-Driven Offline RL? A Bias-Regime Study."*

## Overview

This repository contains a comprehensive implementation of uncertainty-gated reward calibration methods for offline reinforcement learning in recommendation systems. The project investigates how LLM-generated rewards can be leveraged effectively while managing their inherent biases across different regime settings.

## Two Main Pipelines

### 1. Offline RL Pipeline (`reco/`)
- **Algorithm**: Implicit Q-Learning (IQL) with soft conformal gate
- **Encoder**: SASRec (self-attention sequential recommendation)
- **Reward Calibration**: Uncertainty-gated mechanism
- **Datasets**: ML-1M, Yelp, Amazon Video Games
- **Output**: Tables 1 and 2 (main results)
- **Key Features**:
  - Multi-seed driver scripts (seeds: {42, 43, 44} for ML-1M: {42, 7, 123})
  - 5-core preprocessed dataset splits
  - Cached LLM ratings (no API key required for reproduction)
  - Full-rank leave-last-out evaluation harness

### 2. Online RL Baselines Pipeline (`RecSYS-Codes/`)
- **Algorithms**: DQN, PPO, A2C, TRPO
- **Environment**: Frozen SASRec simulator
- **Evaluation**: Sampled-99-negative and full-rank leave-last-out harnesses
- **Output**: Table 3 (cross-paradigm comparison)
- **Random Seeds**: {0, 1, 2}

## Quick Start

### Requirements
- **GPU**: RTX 4070 (12 GB) or equivalent
- **Python**: 3.12
- **PyTorch**: 2.3
- **CUDA**: 12.1
- **All Dependencies**: See `reco/requirements.txt`

### Dependencies
```
torch>=2.2.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
pyyaml>=6.0
tqdm>=4.66
matplotlib>=3.7
seaborn>=0.13
pyarrow>=14.0
```

### 15-Minute Smoke Test

On a single GPU, quickly validate your setup:

```bash
cd reco
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
python scripts/preprocess.py --dataset ml1m
python scripts/run_primary.py --seeds 42 --datasets ml1m \
    --variants R_warm R4_ugmv2_BC
```

**Expected Output**:
- Trains two offline-IQL variants on ML-1M seed 42
- Reports NDCG@10 under full-rank leave-last-out evaluation
- Expected NDCG@10 (paper mean of three seeds):
  - `R_warm`: ~0.049
  - `R4_ugmv2_BC`: ~0.054
- Single-run seed 42 should land within ±0.01 of the three-seed mean

### Full Reproduction

Complete reproduction of all experiments (~15 GPU-hours):

```bash
# Offline RL experiments (Tables 1-2, ~6 GPU-hours)
cd reco
bash scripts/run_all_primary.sh

# Online RL baselines (Table 3, ~9 GPU-hours)
cd ../RecSYS-Codes
python -m src.scripts.run_all_ablations \
    --seeds 0 1 2 --steps 50000

# Generate aggregated result tables
python -m src.scripts.build_result_tables
```

**Note**: Precomputed `simreal_test.json` results are cached in `RecSYS-Codes/results/runs/`, so you can regenerate tables without retraining.

## Repository Structure

### `reco/` — Offline RL Pipeline
```
reco/
├── src/                  # Core modules
│   ├── iql.py           # Implicit Q-Learning implementation
│   ├── conformal_gate.py # Soft conformal uncertainty gating
│   ├── sasrec_encoder.py # SASRec sequential encoder
│   └── eval.py          # Full-rank and sampled evaluation metrics
├── configs/
│   ├── v2.yaml          # Main configuration (datasets, hyperparams)
│   └── v2_grid.yaml     # Hyperparameter sweep configuration
├── scripts/
│   ├── preprocess.py              # Dataset preprocessing (5-core filtering)
│   ├── run_primary.py             # Single experiment runner
│   ├── run_all_primary.sh         # Multi-seed batch runner
│   └── build_result_tables.py     # Result aggregation
├── processed/           # Preprocessed dataset splits (5-core)
├── cache/               # Pre-cached LLM ratings + embeddings
│   └── llm_ratings/     # Dataset-specific LLM reward files
└── requirements.txt     # Python dependencies
```

### `RecSYS-Codes/` — Online RL Baselines
```
RecSYS-Codes/
├── src/
│   ├── agents/          # DQN, PPO, A2C, TRPO implementations
│   ├── env/             # Frozen SASRec simulator environment
│   ├── eval.py          # Multi-harness evaluation
│   └── scripts/
│       ├── run_all_ablations.py       # Ablation runner
│       ├── build_result_tables.py     # Result aggregation
│       └── aggregate_results.py       # CSV aggregation
├── configs/             # Per-dataset YAML configurations
├── results/
│   ├── runs/            # Per-run simulation results (simreal_test.json)
│   └── tables/          # Aggregated CSVs and markdown tables
└── requirements.txt     # Dependencies (subset of reco/)
```

## Configuration Files

### Main Config: `reco/configs/v2.yaml`
Specifies:
- Dataset names and paths
- Model architecture (embedding dims, hidden dims, num layers)
- Training hyperparameters (learning rate, batch size, epochs)
- Evaluation settings (metric types, harness configurations)
- Seed settings for reproducibility

### Sweep Config: `reco/configs/v2_grid.yaml`
Defines hyperparameter ranges for sensitivity analysis:
- Gating temperature
- Reward calibration weights
- Discount factors

## Datasets

| Dataset | Size | Sparsity | Source |
|---------|------|----------|--------|
| ML-1M | 100K users, 1M ratings | ~95% | MovieLens |
| Yelp | 200K users, 5M ratings | ~99% | Yelp Open Dataset 2022 |
| Amazon Video Games | 300K users, 1.3M reviews | ~99.9% | Amazon Product Reviews 2018 |

All datasets preprocessed to 5-core (≥5 interactions per user and item).

## Evaluation Metrics

- **NDCG@K**: Normalized Discounted Cumulative Gain (K ∈ {5, 10})
- **Recall@K**: Fraction of ground-truth items in top-K recommendation
- **Hit Rate**: Probability of ≥1 relevant item in top-K
- **Harness Types**:
  - Full-rank leave-last-out (all items evaluated)
  - Sampled-99-negative (99 random negatives + 1 positive)

## Environment Setup

### Python & CUDA
```bash
# Verify Python version
python --version  # Should be 3.12

# Verify PyTorch & CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Random Seeds
- **Offline IQL**: {42, 43, 44} (ML-1M sign tests also use {42, 7, 123})
- **Online RL**: {0, 1, 2}
- All experiments run with fixed seeds for reproducibility

## Results Summary

### Offline RL (Tables 1-2)
Compare reward calibration variants:
- Baseline warm-start (R_warm)
- Uncertainty-gated variant (R4_ugmv2_BC)
- Cross-dataset consistency on ML-1M, Yelp, Amazon

### Online RL (Table 3)
Benchmark online RL agents (DQN, PPO, A2C, TRPO) against offline IQL to validate cross-paradigm applicability.

## Reproduction Notes

1. **No API keys required**: All LLM ratings are pre-cached in `reco/cache/`
2. **Deterministic results**: Fixed seeds ensure reproducibility
3. **Precomputed results**: `RecSYS-Codes/results/runs/simreal_test.json` enables table regeneration without retraining
4. **GPU memory**: Experiments use ≤12 GB VRAM on RTX 4070

## Troubleshooting

### CUDA Issues
- Verify CUDA 12.1 compatibility: `nvidia-smi`
- Reinstall PyTorch if needed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### Memory Issues
- Reduce batch size in config YAML
- Use gradient accumulation (adjust `accumulation_steps` in config)

### Preprocessing Issues
- Verify dataset files are in correct directory
- Check 5-core filtering logs in `reco/processed/`

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2026uncertainty,
  title={When Does Uncertainty-Gated Reward Calibration Help LLM-Driven Offline RL? A Bias-Regime Study},
  author={Anonymous},
  booktitle={RecSys 2026 (Anonymous Submission)},
  year={2026}
}
```

## License

Code released under the MIT License — see `LICENSE` file for details.

**Datasets**:
- MovieLens 1M: See [MovieLens Terms](https://grouplens.org/datasets/movielens/)
- Yelp Open Dataset 2022: See [Yelp Dataset](https://www.yelp.com/dataset)
- Amazon Product Reviews 2018: See [Amazon Reviews](https://jmcauley.ucsd.edu/data/amazon/)

## Contributing

We welcome contributions and discussions. For bugs or feature requests, please open an issue or pull request.

## Contact

For questions about this research, please refer to the paper or open an issue in this repository.