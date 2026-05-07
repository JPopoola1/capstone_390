# Error Taxonomy

Project: Premier League xG prediction with home/away, travel distance, team/opponent context, and model-search experiments.

Primary target: `xg`  
Primary metric: validation RMSE on the 2024 validation split  
Locked test policy: 2025 season remains untouched for model selection

## Summary

The current loop is dominated by model-specification error: the ridge alpha sweep improved validation RMSE only modestly, then plateaued around `alpha=350`, while the later ridge + ExtraTrees voting model improved further. This suggests the model is not only sensitive to regularization strength; it is also missing nonlinear or interaction-based structure between team, opponent, venue, travel, and season context.

## Error Categories

| Error Type | How It Manifests | Likely Cause | Evidence To Check | Current Status |
|---|---|---|---|---|
| Nonlinear matchup error | Linear ridge performance improves with alpha but plateaus; tree-based or ensemble models perform better | Some xG outcomes depend on interactions between team strength, opponent, venue, season, and travel that a mostly linear model cannot fully capture | Compare residuals from ridge vs tree/ensemble models; inspect matches where ridge misses but ensemble improves | Dominant suspected error |
| High-xG underprediction | Model predicts too low for teams that create unusually many chances in a match | Rare attacking outbursts are hard to predict from fixed effects and travel alone | Sort validation residuals where actual xG is much greater than predicted xG | Needs residual review |
| Low-xG overprediction | Model predicts a decent attacking output for teams that produce very little xG | Team/opponent averages may be too optimistic when game state, tactics, injuries, or matchup suppress attack | Sort validation residuals where predicted xG is much greater than actual xG | Needs residual review |
| Team-style error | Repeated overprediction or underprediction for the same club | Team fixed effects may not capture style changes, manager effects, roster changes, or form within a season | Group validation residuals by team and inspect mean error and RMSE | Likely present |
| Opponent suppression error | Model misses cases where a strong defensive opponent limits xG | Opponent fixed effects may be too coarse or insufficiently interactive with attacking team identity | Group residuals by opponent; inspect low actual xG against strong defensive teams | Likely present |
| Home/away context error | Error patterns differ between home and away matches | Home advantage and travel distance may not fully describe venue effects, rest, tactical setup, or crowd/context effects | Compare residual distributions for home vs away rows | Possible |
| Travel-distance signal error | Adding travel distance does not clearly improve xG prediction and previously worsened the baseline xG RMSE | Travel distance may be weak, confounded with opponent/geography, or less relevant than team/opponent strength | Compare baseline vs travel model RMSE and coefficient stability; check whether distance effect is robust across seasons | Observed in baseline |
| Over-regularization error | RMSE worsens at very high ridge alpha, especially `alpha=1000` | Coefficients are shrunk too aggressively, removing useful signal | Ridge alpha sweep: `alpha=350` RMSE `0.837233`; `alpha=1000` RMSE `0.839931` | Confirmed |
| Under-regularization error | Lower alpha values improve over baseline but do not reach best ridge performance | Model keeps too much variance/noise in coefficients | Ridge alpha sweep: `alpha=10` RMSE `0.838728`; `alpha=350` RMSE `0.837233` | Confirmed |
| Evaluation noise / small validation slice error | Small RMSE differences may look meaningful but could be unstable | Validation is limited to one season, so individual matches can influence close comparisons | Rerun deterministic models; use paired residual comparisons; avoid touching locked 2025 test set | Managed by fixed split and rerun |

## Dominant Error Type

The dominant current error type is nonlinear matchup error.

The controlled ridge alpha sweep shows that changing only the regularization strength improves the model up to about `alpha=350`, but the improvement flattens after that point. The best controlled ridge result is:

- `Ridge(alpha=350)`
- validation RMSE: `0.837233`
- validation R2: `0.116537`

The later model-search history shows that adding a regularized tree component through an even-weight ridge + ExtraTrees voting model improves validation RMSE further to `0.830723`. Because that improvement comes from changing model form rather than only alpha, the likely explanation is that some errors come from nonlinear relationships or interactions that the ridge model cannot represent.

## Practical Interpretation

The model is probably learning broad team, opponent, season, home/away, and travel patterns, but it still struggles when those factors combine in ways that are not additive. In soccer terms, a team's attacking output is not only the sum of team quality, opponent quality, venue, and travel distance. Matchups can behave differently depending on how those factors interact.

## Next Evidence To Collect

1. Generate validation residuals for the best ridge model and the best voting model.
2. Identify the largest underpredictions and overpredictions of xG.
3. Group residuals by team, opponent, and home/away status.
4. Compare where ridge fails but the voting model improves.
5. Use those cases to decide whether the next controlled experiment should vary model class, interaction features, or residual grouping strategy.
