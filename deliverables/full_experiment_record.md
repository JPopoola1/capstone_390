# Full Record of Experiments

The complete experiment record is provided as:

`deliverables/full_experiment_record.tsv`

Source file:

`autoresearch/results.tsv`

The table contains 465 logged AutoResearch experiments with these columns:

| Column | Meaning |
| --- | --- |
| `commit` | Git commit associated with the experiment run |
| `rmse` | Validation RMSE on the 2024 validation split |
| `r2` | Validation R2 on the 2024 validation split |
| `status` | Whether the run was kept, discarded, or used as a baseline |
| `description` | Human-readable experiment description and hyperparameter summary |

Best logged validation result:

| Commit | RMSE | R2 | Status | Description |
| --- | ---: | ---: | --- | --- |
| `01dfedb` | 0.820630 | 0.151229 | keep | maxabs scaler for one hot design |

Selected locked-test model:

| Commit | Validation RMSE | Validation R2 | Status | Description |
| --- | ---: | ---: | --- | --- |
| `9c1d4f3` | 0.821198 | 0.150054 | keep | batch5 tw 78 leaf 17 mf 075 depth 15 n 400 alpha 0.1 seed 42 inter none |

The MaxAbsScaler run is listed as the best logged validation score, but the locked test report identifies the selected final model as the StandardScaler voting model above.
