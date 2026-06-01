# Final Deliverables

This folder contains the separate final submission artifacts derived from `autoresearch/model_performance_report.html` and the project records.

| Required deliverable | File |
| --- | --- |
| Final Report, 4-page NeurIPS-style LaTeX | `final_report_neurips.tex` |
| Local NeurIPS-style wrapper for compilation | `neurips_2024.sty` |
| Cleaned Repository / GitHub link | `cleaned_repository.md` |
| Full record of experiments | `full_experiment_record.tsv`, summarized by `full_experiment_record.md` |
| Final Results Table | `final_results_table.tsv`, readable version in `final_results_table.md` |
| Reflection Memo | `reflection_memo.md` |
| Publication-style figures | `figures/season_mean_xg.svg`, `figures/validation_rmse_comparison.svg` |

Notes:

- The full experiment record contains 465 logged validation experiments.
- The final locked test table contains five repeated runs of the selected model, all with RMSE `0.792404` and R2 `0.064477`.
- The final report is separate from `autoresearch/model_performance_report.html`.
- A PDF was not generated in this workspace because no LaTeX engine was available on PATH; Quarto was present but does not render raw `.tex` files directly.
