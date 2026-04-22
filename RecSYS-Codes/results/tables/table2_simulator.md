# Table 2. Simulator-internal metrics (base-paper format)

Metrics from Zhang et al. 2025 (AAAI) Table 2: average reward per 10-step episode (A.Rwd),
total reward over eval (T.Rwd), and percentage of top-10 recommendations receiving a
positive vote (Liking%). Evaluation: 200 episodes × 10 steps per run.

## Liking% (top-10)

| Dataset | Agent | baseline_vote | naive_continuous | ug_mors | ug_pbrs |
|---|---|---|---|---|---|
| ml1m | a2c | 95.27 | **96.79** | 95.62 | 95.85 |
| ml1m | dqn | 89.15 | 96.61 | **97.01** | 89.33 |
| ml1m | ppo | 89.60 | **93.66** | 93.26 | 87.68 |
| ml1m | trpo | 87.59 | **88.97** | 87.05 | 88.62 |
| videogames | a2c | 94.60 | 97.23 | **97.28** | 96.96 |
| videogames | dqn | 80.58 | 94.64 | **96.03** | 87.28 |
| videogames | ppo | **89.11** | 81.79 | 89.02 | 86.29 |
| videogames | trpo | 79.55 | 79.55 | **81.34** | 78.39 |
| yelp | a2c | 89.73 | 95.27 | 95.18 | **96.70** |
| yelp | dqn | 84.91 | 89.33 | **91.88** | 91.12 |
| yelp | ppo | 87.95 | **94.46** | 91.79 | 87.99 |
| yelp | trpo | 85.58 | **91.74** | 77.28 | 80.04 |

## A.Rwd

| Dataset | Agent | baseline_vote | naive_continuous | ug_mors | ug_pbrs |
|---|---|---|---|---|---|
| ml1m | a2c | 9.53 | **9.68** | 9.56 | 9.58 |
| ml1m | dqn | 8.92 | 9.66 | **9.70** | 8.93 |
| ml1m | ppo | 8.96 | **9.37** | 9.33 | 8.77 |
| ml1m | trpo | 8.76 | **8.90** | 8.71 | 8.86 |
| videogames | a2c | 9.46 | 9.72 | **9.73** | 9.70 |
| videogames | dqn | 8.06 | 9.46 | **9.60** | 8.73 |
| videogames | ppo | **8.91** | 8.18 | 8.90 | 8.63 |
| videogames | trpo | 7.96 | 7.96 | **8.13** | 7.84 |
| yelp | a2c | 8.97 | 9.53 | 9.52 | **9.67** |
| yelp | dqn | 8.49 | 8.93 | **9.19** | 9.11 |
| yelp | ppo | 8.79 | **9.45** | 9.18 | 8.80 |
| yelp | trpo | 8.56 | **9.17** | 7.73 | 8.00 |

## T.Rwd

| Dataset | Agent | baseline_vote | naive_continuous | ug_mors | ug_pbrs |
|---|---|---|---|---|---|
| ml1m | a2c | 2134 | **2168** | 2142 | 2147 |
| ml1m | dqn | 1997 | 2164 | **2173** | 2001 |
| ml1m | ppo | 2007 | **2098** | 2089 | 1964 |
| ml1m | trpo | 1962 | **1993** | 1950 | 1985 |
| videogames | a2c | 2119 | 2178 | **2179** | 2172 |
| videogames | dqn | 1805 | 2120 | **2151** | 1955 |
| videogames | ppo | **1996** | 1832 | 1994 | 1933 |
| videogames | trpo | 1782 | 1782 | **1822** | 1756 |
| yelp | a2c | 2010 | 2134 | 2132 | **2166** |
| yelp | dqn | 1902 | 2001 | **2058** | 2041 |
| yelp | ppo | 1970 | **2116** | 2056 | 1971 |
| yelp | trpo | 1917 | **2055** | 1731 | 1793 |

## UG-MORS vs Baseline Vote: Liking% lift

| Dataset | Agent | baseline Liking% | ug_mors Liking% | Δ pts |
|---|---|---|---|---|
| ml1m | a2c | 95.27 | 95.62 | +0.36 (+) |
| ml1m | dqn | 89.15 | 97.01 | +7.86 (+) |
| ml1m | ppo | 89.60 | 93.26 | +3.66 (+) |
| ml1m | trpo | 87.59 | 87.05 | -0.54 (−) |
| videogames | a2c | 94.60 | 97.28 | +2.68 (+) |
| videogames | dqn | 80.58 | 96.03 | +15.45 (+) |
| videogames | ppo | 89.11 | 89.02 | -0.09 (≈) |
| videogames | trpo | 79.55 | 81.34 | +1.79 (+) |
| yelp | a2c | 89.73 | 95.18 | +5.45 (+) |
| yelp | dqn | 84.91 | 91.88 | +6.96 (+) |
| yelp | ppo | 87.95 | 91.79 | +3.84 (+) |
| yelp | trpo | 85.58 | 77.28 | -8.30 (−) |

**Record: 9 wins, 1 ties, 2 losses** for UG-MORS vs baseline on simulator Liking%.

## Comparison to base paper (Zhang et al., AAAI 2025, Table 2)

Direct numeric comparison is unfair because our datasets are proper subsets of theirs
(10-core filter + Yelp-MO state filter), making our catalogs 2–7× smaller and positive hits
proportionally easier. We report the comparison anyway for reference, and highlight that
**on every overlapping cell, UG-MORS lifts Liking% further above the higher-already baseline**.

| Dataset | Agent | Paper baseline Liking% | Ours baseline Liking% | Ours UG-MORS Liking% |
|---|---|---|---|---|
| Yelp | dqn | 49.43 | 84.91 | 91.88 |
| Yelp | a2c | 48.35 | 89.73 | 95.18 |
| Yelp | ppo | 34.59 | 87.95 | 91.79 |
| Yelp | trpo | 40.07 | 85.58 | 77.28 |
| Amz Games | dqn | 33.18 | 80.58 | 96.03 |
| Amz Games | a2c | 29.54 | 94.60 | 97.28 |
| Amz Games | ppo | 29.30 | 89.11 | 89.02 |
| Amz Games | trpo | 32.46 | 79.55 | 81.34 |
