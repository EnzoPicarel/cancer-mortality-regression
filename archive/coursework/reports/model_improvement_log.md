# Model Improvement Log

## Current selected result

The selected model is the final Ridge + HistGradientBoosting blend, saved as `final_tuned_model.joblib`. Its required Kaggle upload file is now named `submission.csv`; the user reported a Kaggle R² of **0.89414** for these predictions. Older candidate submission CSVs have been removed after their scores were recorded below. Search-result CSVs are retained as experiment evidence.

| Submitted candidate | Kaggle R² |
|---|---:|
| Initial submission (leaderboard entry) | 0.38293 |
| Earlier baseline result reported during tuning | 0.37952 |
| `submission_improved.csv` | 0.73954 |
| `submission_nested.csv` | 0.74609 |
| `submission_advanced.csv` | 0.76054 |
| `submission_feature_search.csv` | 0.78624 |
| `submission_ensemble.csv` | 0.88466 |
| Final tuned blend, now `submission.csv` | **0.89414** |

## Goal and evaluation rules

The previous Kaggle submission scored **0.37952**. I looked for an improvement using only labels in `train.csv`; I did not download or use labels for Kaggle's hidden test rows. The search used a fixed outer split matching the assignment's 70/15/15 split: 1,705 training rows and 366 validation rows formed the 2,071-row development set, while the remaining 366 rows were held out as a local test set. Hyperparameters were selected only by repeated cross-validation within the development set. The local test score was calculated once for the selected configuration.

## Findings that led to the new model

The earlier ordinary least-squares model used the untransformed features and reached local-test R² **0.5964** (MSE **305.06**). A broad cross-validated model search showed that regularization and feature representation mattered much more than adding a deeper neural network:

- Ordinary least squares became numerically unstable after adding encoded state and transformed features.
- Ridge regression with `alpha=1` performed much better by shrinking correlated coefficients.
- Logarithmic versions of the count and scale variables improved the cross-validation score substantially. For the same Ridge search, `state + logs` reached mean R² **0.7839**, while the best `state, no logs` variant reached **0.5568**.
- Adding a state derived from `Geography` helped: `state + logs` reached mean R² **0.7839** versus **0.7598** for `no state + logs`. County names themselves remain excluded because each county occurs once; the extracted state repeats across many counties.
- The `MedianAge` correction is applied consistently: values over 100 are divided by 12 in both files. A separate ablation showed almost no score change from this feature treatment, so it is retained for unit consistency rather than claimed as the source of the score gain.

## Selected model and measured result

The selected configuration is **Ridge regression, alpha = 1**, using:

- Numeric features with median imputation, missingness indicators, and standardization.
- `binnedInc` and the state parsed from `Geography` as one-hot categorical features.
- `log1p` versions of `avgAnnCount`, `avgDeathsPerYear`, `popEst2015`, and `studyPerCap`, alongside the original values.
- The corrected `MedianAge` feature; `id` and county-level `Geography` are excluded.

| Evaluation | R² | MSE |
|---|---:|---:|
| Repeated 5-fold CV on 2,071 development rows (10 folds total) | **0.7839 ± 0.0397** | **169.53 ± 20.62** |
| Untouched 366-row local test | **0.8585** | **106.98** |
| Previous linear model on the same local test rows | 0.5964 | 305.06 |

On the same local test rows, the new model raises R² by **0.2620** and lowers MSE by about **64.9%** relative to the previous linear model. This is a substantial local improvement. It has **not** yet been confirmed on Kaggle; the public leaderboard score is still needed to determine whether it carries over to the competition split.

## Files and next check

The intermediate search-result CSVs and model files named in this historical log were generated during the experiments and pruned from the folder on 2026-09-29. Their score and selection details are preserved below; the experiment scripts remain available to regenerate the outputs. The selected model weights are `final_tuned_model.joblib`.

- `ridge_tuning.py` — reproducible repeated-CV selection, one-time local holdout evaluation, final refit, and submission generation.
- `ridge_cv_search.csv` — all Ridge alpha and feature-variant CV results.
- `ridge_best_holdout.csv` — selected configuration and untouched local-test result.
- `ridge_best_model.joblib` — final Ridge pipeline plus configuration and evaluation metadata.
- `submission_improved.csv` — 610 predictions; columns and IDs were checked against `test.csv`, and every prediction is finite. The prediction range is 91.87–255.86.

The first improved candidate, `submission_improved.csv`, was submitted and scored **0.73954**, compared with the previous score of **0.37952**. This is the current Kaggle benchmark. If later candidates score substantially below their local estimates, investigate train/test distribution differences and the competition split; do not optimize against hidden test labels.

## Round 2: nonlinear interactions

After the Ridge submission scored **0.73954** on Kaggle, I tested spline Ridge, target-log Ridge, and pairwise-interaction Ridge. The most promising candidate is a Ridge model with alpha **0.3**, state and count-log features, and pairwise interactions among 14 numeric features. Its repeated 5-fold CV on the development set reached R² **0.8317 ± 0.0162**; the previously used local split scored R² **0.8929** (MSE **80.96**), compared with **0.8585** for the first Ridge candidate.

There is a validation caveat: the 14 interaction variables were chosen after looking at correlations with `TARGET_deathRate` over the full training file. Although the hidden Kaggle labels were never used and the prediction code fits only on `train.csv`, this makes the CV and local split scores optimistic as estimates of generalization. As a check, I also ran a less selective interaction model that chooses numeric features within each CV fold; it reached only R² **0.7256 ± 0.0855** and scored **0.5621** on the reused local split. So the manually chosen interaction model is a promising Kaggle experiment, but its local numbers do not guarantee a higher public score.

The candidate file is `submission_advanced.csv`. It has 610 rows, the expected columns, matching IDs, and finite predictions. Submit it as one new Kaggle version to measure whether its extra interactions help on the competition set; keep `submission_improved.csv` as the current **0.73954** benchmark until the new public score confirms an improvement. The fitted model and search results are `advanced_best_model.joblib`, `advanced_cv_search.csv`, and `nested_interaction_cv.csv`.

## Round 3: population and incidence rate features

The user reported these public Kaggle results:

| Candidate | Kaggle R² |
|---|---:|
| `submission_improved.csv` | 0.73954 |
| `submission_nested.csv` | 0.74609 |
| `submission_advanced.csv` | **0.76054** |

I tested rate features motivated by how county cancer counts relate to population: `avgDeathsPerYear / avgAnnCount`, annual cases per 100,000 residents, and annual deaths per 100,000 residents. These use only observed predictors in each row; no hidden target labels were used. In repeated 5-fold CV (3 repeats, 15 validation folds) on the 2,437 labeled rows, the best Ridge model with these rates and alpha **0.75** reached R² **0.8622 ± 0.0145**, versus **0.8072 ± 0.0220** for the state plus log-count baseline on the same folds. The mean CV MSE was **108.67** versus **151.86**. The pairwise interaction version was weaker (best R² **0.8485 ± 0.0619**), so it was not selected.

The selected model and predictions are in `feature_search_model.joblib` and `submission_feature_search.csv`; the full comparison is in `feature_search_results.csv`, with code in `feature_search.py`. The generated submission has 610 rows, the expected `id` and `TARGET_deathRate` columns, IDs in the same order as `test.csv`, finite predictions, and a prediction range of 100.61–262.76.

This repeated CV result is a strong reason to try the candidate, but it is not a Kaggle score. Also, the rate feature set was chosen after comparing several variants on these same labeled rows, so 0.8622 may be optimistic. The public Kaggle score is the decisive comparison. Keep `submission_advanced.csv` as the best confirmed submission at **0.76054** unless Kaggle confirms a higher score for the new file.

## Round 4: blend linear and nonlinear models

The user reported that `submission_feature_search.csv` scored **0.78624** on Kaggle. I then compared Ridge and tree boosting using the same 15 repeated validation folds and the same rate features. The strongest individual tree model was HistGradientBoosting with 15 leaves (mean fold R² **0.8821 ± 0.0215**). A blend of **40% Ridge (alpha 0.75) + 60% HistGradientBoosting (15 leaves)** reached mean fold R² **0.8966**; across the three complete CV repeats, the standard deviation of the aggregate R² was **0.0026**. Ridge alone scored **0.8572 ± 0.0281** on these exact folds. This is a material, same-fold improvement over either earlier candidate's validation result, though it is still an internal estimate and the Kaggle leaderboard must confirm generalization.

The new candidate is `submission_ensemble.csv`; its reproducible search and refit code is `ensemble_search.py`, with fold scores in `ensemble_search_results.csv`. The model uses train labels only and excludes county names and row ID. Predictions were fitted from all 2,437 labeled rows; the 610-row submission has IDs matching `test.csv`, finite values, and range 109.03–251.79. Submit this candidate as a new Kaggle version and compare against **0.78624**, the current best confirmed score. Do not overwrite that benchmark until Kaggle reports the new result.

The user reported that `submission_ensemble.csv` scored **0.88466** on Kaggle, which is now the best confirmed score. A final focused search varied HistGradientBoosting leaf count, learning rate, iteration count and minimum leaf size; Ridge alpha; and blend weight, using the same repeated CV folds as Round 4. The best combination was Ridge alpha **0.75** at **35%** plus HistGradientBoosting (15 leaves, minimum leaf size 12) at 65%, with repeated-CV R² **0.8986**. This is only **0.0020** above the previous blend's 0.8966, after comparing multiple configurations, so the apparent gain is too small to treat as a reliable improvement.

At the time of this search, the tuned alternative had only a marginal CV advantage, so `submission_ensemble.csv` remained the confirmed best at **0.88466**. `final_tuned_model.joblib`, `final_tuning.py`, `final_tuning_individual.csv`, and `final_tuning_blends.csv` record the tuned configuration and search. The subsequent Kaggle result is recorded below and in the score table above.

The user submitted the final tuned blend and reported a Kaggle score of **0.89414**, an improvement of **0.00948** over the previous confirmed 0.88466. This is the best confirmed competition result. Its canonical prediction filename is now `submission.csv`, and the model weights are in `final_tuned_model.joblib`. The full homework completion checklist is in `submission_checklist.md`.
