# Full v2 ablation grid — 3 datasets x 5 rewards x 5 seeds = 75 runs.
# Budget: ~10min/run on an RTX 4070 => ~12.5 hours.
#
# Ablation rungs (the "why each piece matters" story):
#   R0. IQL + naive_continuous          (reward but no gate, no BC anchor)
#   R1. IQL + binary                    (sparse reward baseline)
#   R2. IQL + hard_gate                 (step gate, no BC anchor)
#   R3. IQL + ug_mors_v2 (no BC)        (soft CRC gate, gate_complement=uniform)
#   R4. IQL + ug_mors_v2 + BC           (our method)
#   R5. IQL + ug_mors_v2 + BC + pess    (our method + pessimism)
#
# Usage: from the repo root:
#     .\scripts\run_all_ablations_v2.ps1

$ErrorActionPreference = "Stop"
$root = (Get-Item -Path ".\").FullName
$datasets = @("movielens-1m", "amazon-videogames", "yelp")
$seeds = @(42, 43, 44, 45, 46)
$rewards = @("naive_continuous", "binary", "hard_gate", "ug_mors_v2", "ug_mors_v2", "ug_mors_v2")
$rung_tags = @("R0_naive", "R1_binary", "R2_hardgate", "R3_ugmv2_noBC", "R4_ugmv2_BC", "R5_ugmv2_BC_pess")
$rung_overrides = @(
  @("loss.bc_weight_mode=`"uniform`"", "loss.pessimism_lambda=0.0"),
  @("loss.bc_weight_mode=`"uniform`"", "loss.pessimism_lambda=0.0"),
  @("loss.bc_weight_mode=`"uniform`"", "loss.pessimism_lambda=0.0"),
  @("loss.bc_weight_mode=`"uniform`"", "loss.pessimism_lambda=0.0"),
  @("loss.bc_weight_mode=`"gate_complement`"", "loss.pessimism_lambda=0.0"),
  @("loss.bc_weight_mode=`"gate_complement`"", "loss.pessimism_lambda=0.1")
)

foreach ($ds in $datasets) {
  for ($i = 0; $i -lt $rewards.Count; $i++) {
    $rw = $rewards[$i]
    $tag = $rung_tags[$i]
    $extra = $rung_overrides[$i]
    foreach ($seed in $seeds) {
      $out = Join-Path $root "runs_v2\$ds\$tag\seed$seed"
      if (Test-Path (Join-Path $out "final.json")) {
        Write-Host "[skip] already done: $out"
        continue
      }
      $overrides = @(
        "dataset.name=`"$ds`"",
        "reward.name=`"$rw`"",
        "seed=$seed"
      ) + $extra
      Write-Host "[run] $ds / $tag / seed=$seed"
      py -3.12 scripts/run_experiment_v2.py `
        --config configs/v2.yaml `
        --override @overrides `
        --out $out
    }
  }
}

Write-Host "`nAll v2 ablations complete. Run make_paper_figures.py next."
