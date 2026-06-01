# Final Results Table

Locked model: `batch5 tw 78 leaf 17 mf 075 depth 15 n 400 alpha 0.1 seed 42 inter none`

Testing set: 760 rows from the 2025 season, dated 2024-08-16 through 2025-05-25.

| Run | Train rows | Test rows | RMSE | R2 | Train time seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2280 | 760 | 0.792404 | 0.064477 | 0.970434 |
| 2 | 2280 | 760 | 0.792404 | 0.064477 | 0.950336 |
| 3 | 2280 | 760 | 0.792404 | 0.064477 | 0.948985 |
| 4 | 2280 | 760 | 0.792404 | 0.064477 | 0.951640 |
| 5 | 2280 | 760 | 0.792404 | 0.064477 | 0.970791 |

Summary: the repeated locked-test RMSE was `0.792404` and the repeated locked-test R2 was `0.064477`.

Validation context: the selected model's validation RMSE was `0.821198` with validation R2 `0.150054`. The best logged validation run was `maxabs scaler for one hot design` with RMSE `0.820630`, but `autoresearch/model.py` was restored to the selected StandardScaler model before the locked final test report.
