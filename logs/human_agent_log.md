# Human / Agent Activity Log

## Human did

- Defined the research question, success criterion, and week-2 deliverables.
- Supplied the Premier League match dataset and the travel-distance-enhanced dataset.
- Specified that the workflow must be reproducible with a fixed metric and a locked test set.
- Linked the `capstone_390` GitHub repository and updated the dataset files there.

## Agent did

- Downloaded the current repository contents into the local workspace.
- Created a reproducible Python project layout around the existing repo files.
- Copied the original and distance-enhanced datasets into `data/raw/`.
- Implemented a chronological train/validation/test split and saved the split manifest.
- Built an end-to-end baseline comparing:
  - home/away plus fixed effects
  - home/away plus travel distance plus fixed effects
- Added a fixed validation metric, runtime budget artifact, experiment log, and locked test plan.
- Added README instructions so the baseline can be rerun consistently.
