# Locked Test Set Plan

The locked test set is the full 2025 Premier League season.

## Rules

- Do not use 2025 rows for model fitting.
- Do not use 2025 rows for feature engineering decisions.
- Do not use 2025 rows for model selection.
- Do not use 2025 rows when choosing transformations, thresholds, or hyperparameters.
- Use the 2024 validation season for iterative model comparisons.
- Touch the 2025 test season only for the final evaluation of a selected model.

## Current split

- Train: seasons 2021-2023
- Validation: season 2024
- Locked test: season 2025
