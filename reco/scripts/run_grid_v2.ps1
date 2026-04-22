# Final v2 grid: 26 runs total, ~41 hr on RTX 4070 (assumes bf16 + eval_every=2).
#
# Structure (locked 2026-04-17):
#   ML-1M     : {R_warm, R0, R2, R3, R4, R5} x {42, 7, 123}  = 18 runs  (~23 hr)
#   Amazon-VG : {R_warm, R0, R2, R3, R4, R5} x {42}          =  6 runs  (~14 hr)
#   Yelp      : {R_warm, R4}                x {42}          =  2 runs  (~ 4 hr)
#
# Rungs:
#   R_warm  supervised SASRec only (epochs=1, steps=0) — "zero-RL" baseline
#   R0      IQL + naive_continuous      (reward but no gate, no BC)
#   R2      IQL + hard_gate             (step gate, no BC)
#   R3      IQL + ug_mors_v2 (no BC)    (soft CRC gate, bc_weight_mode=uniform)
#   R4      IQL + ug_mors_v2 + BC       (our main method)
#   R5      IQL + ug_mors_v2 + BC + pess (our main method + pessimism)
#
# Hyperparameters are globally locked in configs/v2_grid.yaml — no per-dataset tuning.
# MAE_highU / MSE_highU / corr(|gap|,u_epi) are seed-deterministic under leave-last-out,
# so seed variance is reported for policy metrics (NDCG/HR/TailHR) only.
#
# Usage:  .\scripts\run_grid_v2.ps1

$ErrorActionPreference = "Stop"
$root = (Get-Item -Path ".\").FullName

# rung -> (reward, bc_mode, pess_lambda, extra_overrides)
$rungs = @(
  @{ tag="R_warm";   reward="ug_mors_v2";       bc="gate_complement"; pess=0.0; extra=@("agent.epochs=1", "agent.steps_per_epoch=0", "agent.warmup_steps=7500") },
  @{ tag="R0_naive"; reward="naive_continuous"; bc="uniform";         pess=0.0; extra=@() },
  @{ tag="R2_hardgate";     reward="hard_gate";     bc="uniform";         pess=0.0; extra=@() },
  @{ tag="R3_ugmv2_noBC";   reward="ug_mors_v2";    bc="uniform";         pess=0.0; extra=@() },
  @{ tag="R4_ugmv2_BC";     reward="ug_mors_v2";    bc="gate_complement"; pess=0.0; extra=@() },
  @{ tag="R5_ugmv2_BC_pess";reward="ug_mors_v2";    bc="gate_complement"; pess=0.1; extra=@() }
)

# dataset -> list of seeds to run (ML-1M gets 3 for policy-metric variance; others 1)
$plan = @(
  @{ ds="movielens-1m";     seeds=@(42, 7, 123); rung_filter=$null },   # all 6 rungs
  @{ ds="amazon-videogames";seeds=@(42);         rung_filter=$null },   # all 6 rungs
  @{ ds="yelp";             seeds=@(42);         rung_filter=@("R_warm", "R4_ugmv2_BC") }   # only R_warm + R4
)

$total = 0
foreach ($p in $plan) {
  $n = $p.seeds.Count * ($(if ($p.rung_filter) { $p.rung_filter.Count } else { $rungs.Count }))
  $total += $n
}
Write-Host "[grid] total runs planned: $total"

$done = 0; $skipped = 0
foreach ($p in $plan) {
  $ds = $p.ds
  foreach ($rung in $rungs) {
    if ($p.rung_filter -and ($p.rung_filter -notcontains $rung.tag)) { continue }
    foreach ($seed in $p.seeds) {
      $out = Join-Path $root "runs_v2\grid\$ds\$($rung.tag)\seed$seed"
      $finalPath = Join-Path $out "final.json"
      if (Test-Path $finalPath) {
        # Check for the _stale marker (used to invalidate pre-patch runs
        # without deleting the file, in sandboxed environments where
        # delete is blocked). Any final.json containing {"_stale": true}
        # is treated as NOT done and the run is re-executed.
        $stale = $false
        try {
          $probe = Get-Content -Raw -Path $finalPath | ConvertFrom-Json
          if ($null -ne $probe._stale -and $probe._stale) { $stale = $true }
        } catch { }
        if (-not $stale) {
          Write-Host "[skip] $ds / $($rung.tag) / seed=$seed (already done)"
          $skipped++; continue
        } else {
          Write-Host "[rerun] $ds / $($rung.tag) / seed=$seed (stale marker detected)"
        }
      }
      $overrides = @(
        "dataset.name=`"$ds`"",
        "reward.name=`"$($rung.reward)`"",
        "loss.bc_weight_mode=`"$($rung.bc)`"",
        "loss.pessimism_lambda=$($rung.pess)",
        "seed=$seed"
      ) + $rung.extra
      Write-Host "`n[run $($done+1)/$total] $ds / $($rung.tag) / seed=$seed"
      py -3.12 scripts/run_experiment_v2.py `
        --config configs/v2_grid.yaml `
        --override @overrides `
        --out $out
      $done++
    }
  }
}
Write-Host "`n[grid] complete: $done run, $skipped skipped (resumed)."
