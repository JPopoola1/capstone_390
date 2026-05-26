# Batch 2 Experiment Comparison

Compared the 20 new `batch2` runs against the 20 most recent rows that existed in `results.tsv` before this batch.

Summary:

| Metric | Previous 20 | New 20 | Delta |
| --- | ---: | ---: | ---: |
| Average RMSE | 0.830260 | 0.827387 | -0.002873 |
| Best RMSE | 0.828508 | 0.825252 | -0.003256 |

Best previous run: `ridge-focused tree_depth_12` (`rmse=0.828508`)

Best new run: `batch2 tree weight 066 leaf 18 mf 075` (`rmse=0.825252`)

| # | Previous description | Previous RMSE | Previous status | New description | New RMSE | New status | RMSE delta | R2 delta |
| ---: | --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| 1 | ridge-focused tree_depth_8 | 0.832287 | discard | batch2 constants baseline ridge extra trees | 0.828508 | discard | -0.003779 | 0.007909 |
| 2 | ridge-focused tree_depth_9 | 0.831308 | discard | batch2 tree weight 048 | 0.828667 | discard | -0.002641 | 0.005527 |
| 3 | ridge-focused tree_depth_11 | 0.829051 | keep | batch2 tree weight 052 | 0.828371 | keep | -0.000680 | 0.001422 |
| 4 | ridge-focused tree_depth_12 | 0.828508 | keep | batch2 tree weight 054 | 0.828255 | keep | -0.000253 | 0.000529 |
| 5 | ridge-focused tree_mf_50 | 0.831620 | discard | batch2 tree weight 056 | 0.828160 | keep | -0.003460 | 0.007236 |
| 6 | ridge-focused tree_mf_60 | 0.831372 | discard | batch2 tree weight 058 | 0.828087 | keep | -0.003285 | 0.006871 |
| 7 | ridge-focused tree_mf_80 | 0.829948 | discard | batch2 tree weight 060 | 0.828036 | keep | -0.001912 | 0.003997 |
| 8 | ridge-focused tree_mf_90 | 0.829940 | discard | batch2 tree weight 062 | 0.828005 | keep | -0.001935 | 0.004043 |
| 9 | ridge-focused tree_n_250 | 0.829718 | discard | batch2 tree weight 064 | 0.827996 | keep | -0.001722 | 0.003596 |
| 10 | ridge-focused tree_n_600 | 0.830219 | discard | batch2 tree weight 066 | 0.828009 | discard | -0.002210 | 0.004618 |
| 11 | ridge-focused robust_base | 0.828571 | discard | batch2 tree weight 064 leaf 14 | 0.827533 | keep | -0.001038 | 0.002168 |
| 12 | ridge-focused w_45_leaf_12 | 0.830229 | discard | batch2 tree weight 064 leaf 13 | 0.827533 | discard | -0.002696 | 0.005633 |
| 13 | ridge-focused w_45_leaf_18 | 0.829651 | discard | batch2 tree weight 064 leaf 12 | 0.827541 | discard | -0.002110 | 0.004407 |
| 14 | ridge-focused w_55_leaf_12 | 0.830444 | discard | batch2 tree weight 064 leaf 16 | 0.827345 | keep | -0.003099 | 0.006476 |
| 15 | ridge-focused w_55_leaf_18 | 0.829994 | discard | batch2 tree weight 064 leaf 18 | 0.825859 | keep | -0.004135 | 0.008630 |
| 16 | ridge-focused w_60_mf_60 | 0.831739 | discard | batch2 tree weight 064 leaf 20 | 0.826835 | discard | -0.004904 | 0.010253 |
| 17 | ridge-focused w_60_mf_80 | 0.830480 | discard | batch2 tree weight 062 leaf 18 | 0.825927 | discard | -0.004553 | 0.009504 |
| 18 | ridge-focused w_40_mf_60 | 0.831571 | discard | batch2 tree weight 066 leaf 18 | 0.825813 | keep | -0.005758 | 0.012029 |
| 19 | ridge-focused w_40_mf_80 | 0.830044 | discard | batch2 tree weight 066 leaf 18 mf 075 | 0.825252 | keep | -0.004792 | 0.009996 |
| 20 | final ridge extra trees depth 12 | 0.828508 | discard | batch2 tree weight 066 leaf 18 mf 080 | 0.826001 | discard | -0.002507 | 0.005228 |
