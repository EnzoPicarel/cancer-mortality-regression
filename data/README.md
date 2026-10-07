# Dataset

Place the course competition `train.csv` and `test.csv` in this directory,
or supply an explicit path to each command. Real competition files are not
redistributed in this portfolio package and are ignored by Git.

The local coursework files used for the recorded replay contained:

- `train.csv`: 2,437 rows, 35 columns including `id` and `TARGET_deathRate`.
- `test.csv`: 610 rows, 34 columns, no target labels.

These are aggregated county-level observations. The course attributes their
underlying sources to the American Community Survey, clinicaltrials.gov and
cancer.gov. The precise vintage and preprocessing of the supplied competition
export were not independently reconstructed. No public download URL is claimed.

`Geography` identifies a county and state. County identity and row IDs are excluded
from predictors; enhanced models extract the repeated state label. This does not
make a random county split equivalent to evaluation on unseen states or future years.

Features associated with deaths may use information measured during the target
period. The no-deaths ablation removes `avgDeathsPerYear` and every derived feature
that uses it. It does not establish that all remaining predictors are prospectively
available. `incidenceRate` and annual case counts remain included.

The `MedianAge > 100` correction divides apparent month-valued observations by
12, following the documented coursework hypothesis. It is an explicit data rule,
not a learned transformation or a verified correction to the original data source.

Run `python demo.py` for synthetic data that exercise the pipeline without the
competition files. Synthetic metrics are never used as benchmark claims.
