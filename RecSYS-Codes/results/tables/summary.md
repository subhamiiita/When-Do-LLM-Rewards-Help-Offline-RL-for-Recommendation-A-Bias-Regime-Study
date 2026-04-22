# Summary — UG-MORS vs Baseline Vote

Per (dataset, agent), lift in NDCG@10 of `ug_mors` over `baseline_vote`:

| Dataset | Agent | baseline NDCG | ug_mors NDCG | Δ (abs) | Δ (rel) |
|---|---|---|---|---|---|
| ml1m | a2c | 0.0765 | 0.1649 | +0.0884 | +115.5% |
| ml1m | dqn | 0.0547 | 0.2901 | +0.2354 | +430.6% |
| ml1m | ppo | 0.0491 | 0.1902 | +0.1411 | +287.6% |
| ml1m | trpo | 0.0453 | 0.0663 | +0.0210 | +46.5% |
| videogames | a2c | 0.0555 | 0.0847 | +0.0292 | +52.5% |
| videogames | dqn | 0.0443 | 0.0899 | +0.0456 | +102.9% |
| videogames | ppo | 0.0572 | 0.0696 | +0.0124 | +21.7% |
| videogames | trpo | 0.0465 | 0.0471 | +0.0005 | +1.2% |
| yelp | a2c | 0.0545 | 0.1532 | +0.0987 | +181.0% |
| yelp | dqn | 0.0475 | 0.1519 | +0.1044 | +219.5% |
| yelp | ppo | 0.0556 | 0.1207 | +0.0651 | +117.0% |
| yelp | trpo | 0.0437 | 0.0599 | +0.0161 | +36.9% |

UG-MORS beats baseline on **12/12** (dataset, agent) combinations.
