# capstone_390

Quantify how important playing at home versus away is, including the travel distance to the away stadium. The main outcome is team attacking performance measured by expected goals (`xG`).

This repository now includes a reproducible week-2 baseline for the research question:

How does travel distance affect a soccer team's offensive and defensive performance when measured by expected goals?

## Repository contents

- `final_matches.csv`: original match-level dataset at the repo root
- `final_matches_distance.csv`: match-level dataset with away travel distance at the repo root
- `data/raw/`: copies of both datasets for reproducible scripts
- `scripts/run_baseline.py`: end-to-end baseline runner
- `src/baseline.py`: data prep, split logic, linear baselines, and significance testing
- `data/splits/split_manifest.csv`: fixed split assignment written by the baseline run
- `outputs/baseline_metrics.json`: saved validation metrics and coefficient summaries
- `outputs/runtime_budget.json`: measured runtime against the fixed budget
- `logs/experiment_log.md`: first experiment log entry
- `logs/human_agent_log.md`: record of what the human did and what the agent did
- `LOCKED_TEST_PLAN.md`: locked test set policy

## Research framing

- Primary target: `xg`
- Secondary target: `xga`
- Primary metric: validation RMSE on `xg`
- Success criterion: adding travel distance to the baseline should create a statistically significant improvement in predicting `xg`

## Fixed split policy

To avoid leakage from future seasons, this project uses a chronological split:

- Train: 2021-2023
- Validation: 2024
- Locked test: 2025

The 2025 season is not used for fitting or model selection in the current baseline.

## Current baseline

Two ordinary least squares baselines are fit for both `xg` and `xga`:

1. `baseline_home_away`
   - intercept
   - home/away indicator
   - team fixed effects
   - opponent fixed effects
   - season fixed effects
2. `home_away_plus_travel`
   - all baseline terms
   - away travel distance in 100-mile units

## Reproducible run instructions

### Bundled Python runtime used in this workspace

```powershell
& "C:\Users\jorda\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\scripts\run_baseline.py
```

### Standard Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python .\scripts\run_baseline.py
```

Run commands from the repository root.

## Current baseline result

From the latest run on the fixed 2024 validation season:

- xG baseline RMSE: `0.9645`
- xG travel-model RMSE: `0.9696`
- xG result: adding travel worsened the primary validation metric in this first baseline
- estimated home effect on xG: `+0.1778`
- estimated travel effect on xG per 100 away miles: `-0.0452`
- xGA baseline RMSE: `0.9958`
- xGA travel-model RMSE: `0.9880`
- runtime: about `0.30` seconds with a fixed `60` second budget

See `outputs/baseline_metrics.json` for the full saved metrics.

## Week-2 deliverables

- Working baseline: complete
- Working end-to-end baseline run: complete
- Fixed metric: complete
- Fixed validation metric: complete
- Locked test set plan: complete
- README with reproducible instructions: complete
- First experiment log entry: complete
- Runtime budget for one iteration: complete

## GitHub note

The project files are prepared locally inside this repository snapshot. If you want the remote GitHub repo updated from this environment, I’ll need either a working local `git` setup in the thread or a workflow you want me to follow for uploading changes.
