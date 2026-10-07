# Verification record

Verified on 7 October 2026 with Python 3.12.3, NumPy 1.26.4, pandas 2.1.4,
scikit-learn 1.4.1.post1, Joblib 1.3.2 and Matplotlib 3.6.3.
Core modeling dependencies are pinned to the recorded versions; Matplotlib has
a compatible range so installation can choose a wheel for the Python version.
The optional CatBoost/LightGBM archive dependencies are not a verified lockfile.

## Executed checks

- Six focused unit tests pass, including training-only imputation statistics,
  disjoint partitions, no-deaths input exclusion, checkpoint inference,
  unseen categories, required input schema, and non-mutating data correction.
- The complete synthetic demo trains six candidates, reloads a saved estimator
  and writes a 20-row prediction file. Its metrics are synthetic only.
- Full real-data replay completed for all six fixed candidates: 10 CV folds
  per model, validation selection, development refit and final comparison.
- The selected candidate is `blend`, chosen by validation R² before test scoring.
- Separate all-label `blend` refit completed on 2,437 observations.
- Prediction export completed for the 610 unlabeled county rows. Column names,
  finite predictions, row count and exact ID order were checked.

## Reproduction

Place the original course CSVs in `data/`, then run from the project directory:

```bash
python -m unittest discover -s tests -v
python demo.py --output artifacts/demo-check
python train.py --data data/train.csv --output artifacts/replay-check
python refit.py --data data/train.csv --model blend --output artifacts/refit-check.joblib
python predict.py --data data/test.csv --checkpoint artifacts/refit-check.joblib --output artifacts/submission-check.csv
```

Fresh output paths preserve prior evidence. `results/metadata.json` records the
input fingerprint, software versions and row indices for the committed replay;
`results/cv_folds.csv` contains its 60 fold results. The figure and README table
derive from `results/evaluation.csv`.

## Interpretation

The `linear` validation result reproduces the original 70%-fit baseline to
rounding. Its final test value differs from the coursework report because this
edition refits every fixed candidate on 85% of the labels before test scoring.
Historical neural-network comparisons remain in the archive.

The enhanced ensemble's test R² is 0.9149; the same blend with death counts and
their derived ratios/logs removed reaches 0.6412. This ablation changes only the
feature set, keeping model settings, split and refit protocol fixed. Case counts
and incidence remain predictors. The result demonstrates sensitivity to those
features, not proof of causal effects or availability before an outcome occurs.

These data were used in prior feature and model development. The new pipeline
isolates fitting within its splits, but the replay is not historically blind.
The CV fold standard deviation is descriptive, not a confidence interval.
The separately reported historical Kaggle scores were not verified directly
against the leaderboard during this work.
