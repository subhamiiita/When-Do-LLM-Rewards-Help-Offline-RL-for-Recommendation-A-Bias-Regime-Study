# Run the full ablation grid for the paper: 3 datasets x 2 agents x 4 rewards.
$ErrorActionPreference = "Stop"
$datasets = @("movielens-1m", "amazon-videogames", "yelp")
$agents   = @("dqn", "ppo")
$rewards  = @("binary", "naive_continuous", "hard_gate", "ug_mors")

foreach ($d in $datasets) {
  foreach ($a in $agents) {
    foreach ($r in $rewards) {
      $tag = "$d-$a-$r"
      Write-Host "===== $tag ====="
      py -3.12 scripts/run_experiment.py `
          --config configs/base.yaml `
          --override "dataset.name=`"$d`"" "agent.name=`"$a`"" "reward.name=`"$r`"" `
          --out "runs/$tag"
    }
  }
}
Write-Host "all runs complete"
