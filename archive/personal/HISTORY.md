# Personal Kaggle experiments

Separate sandbox for personal leaderboard experiments. The submitted homework folder and archive are left unchanged.

## Baseline

- Kaggle score reported by the user: **R² = 0.89414**.
- Model: 35% Ridge + 65% HistGradientBoosting.

## Experiments so far

- CatBoost alone: repeated 5-fold CV (3 repeats) R² **0.89532**.
- LightGBM alone: repeated 5-fold CV (3 repeats) R² **0.88528**.
- Current blend on the same folds: R² **0.89861**.
- Extended features and a 40% CatBoost blend: CV R² **0.90148**. This is a modest internal gain, not yet confirmed on Kaggle.

The user submitted `submissions/submission_extended_blend.csv` and reported Kaggle R² **0.91231**.

### Exact-pipeline CatBoost refinement

The refinement used the exact model pipeline that generated the 0.91231 submission as its out-of-fold reference, then compared additional CatBoost models at 2,500, 3,500, and 5,000 iterations on identical repeated 5-fold splits (3 repeats). The best CV blend adds depth-5 CatBoost (learning rate 0.03, L2 7, 5,000 iterations) at 45% to the existing blend at 55%:

- Exact current pipeline: CV R² **0.90104 ± 0.00239**.
- Refined blend: CV R² **0.90290 ± 0.00099**.

This is a modest internal gain, not a guaranteed Kaggle gain. The ready-to-submit candidate is `submissions/submission_refined_catboost5000.csv` (610 rows, ordered test IDs, finite predictions). Its fitted model and reproduction script are `models/personal_refined_catboost5000.joblib` and `code/make_refined_submission.py`. The previous 0.91231 submission remains available as `submissions/submission_extended_blend.csv`.

The full-target lookup was evaluated only as a diagnostic and removed; it is not part of the model or any candidate submission. The experiments use the provided training labels only.

## Folders

- `code/`: model and feature experiments
- `data/`: competition train/test files
- `models/`: saved fitted models
- `results/`: cross-validation comparisons
- `submissions/`: baseline and personal candidate CSVs
- `vendor/`: isolated CatBoost and LightGBM packages
