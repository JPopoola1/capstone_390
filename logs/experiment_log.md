# Experiment Log

## 2026-04-22 - Baseline v1

- Objective: predict team xG and xGA without using the locked 2025 test set.
- Split: train=2021-2023, validation=2024, test=2025 locked.
- Primary metric: validation RMSE on xG.
- Baseline RMSE (home/away only plus team/opponent/season fixed effects): 0.9645.
- Travel RMSE (baseline + away travel distance): 0.9696.
- Travel feature result on xG: worsens.
- Paired permutation p-value for the xG loss difference: 0.0002.
- Estimated home effect on xG: 0.1778.
- Estimated travel effect on xG per 100 away miles: -0.0452.
- Runtime: 0.32 seconds.