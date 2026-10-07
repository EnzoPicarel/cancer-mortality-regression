# Coursework and competition history

This archive records the work preceding the supported root-level portfolio CLI.

- `coursework/code/`: original linear/MLP comparisons, loss experiments,
  feature searches, Ridge tuning and blending. These scripts expect `data/`,
  `results/`, `models/`, and sometimes `deliverables/` under their parent folder.
- `coursework/reports/`: historical homework and improvement reports, retained
  as records rather than updated benchmark claims. Intermediate outputs were
  pruned during the original work; not every historical score has a surviving CSV.
- `coursework/results/`: the surviving model, loss and age-ablation tables.
- `personal/code/`: subsequent CatBoost/LightGBM experiments. The only adaptation
  in these recovered sources removes local `vendor/` path injection. Install
  `requirements-experiments.txt` if exploring them. Their experiments and directory
  assumptions have not been revalidated end to end in this preparation.
- `personal/results/`: surviving cross-validation comparisons.
- `personal/HISTORY.md`: recorded competition scores and candidate descriptions.

The supported and tested commands are `train.py`, `refit.py`, `predict.py` and
`demo.py` at the project root. Archived scripts are evidence of experiments,
not additional supported entrypoints. Real data, original weight files,
instructor handouts, personal prompt transcripts and installed third-party
packages are not included in this portfolio package.

## Three distinct protocols

1. **Original architecture comparison:** fixed 70/15/15 split; models fitted on
   the 70% train split, selected using validation, then test results reported.
2. **Competition optimization:** later searches used all supplied labels and
   repeated cross-validation; selected settings were refitted on every labeled
   row to produce predictions for the unlabeled competition file. Several
   experiments reused earlier splits and feature-selection feedback.
3. **Portfolio replay:** fixed previously developed candidates, preprocessing
   fitted inside each training fold, validation-only candidate selection,
   then refit on train+validation for the final comparison. This is a replay on
   historically used data, not a newly acquired independent or prospective test.

Historical Kaggle R² scores of **0.89414** and **0.91231** were reported by Enzo
and recorded in the project documents. They have not been independently
verified against the leaderboard during this preparation. The 5,000-iteration
CatBoost refinement has a recorded CV gain but no recorded new Kaggle score.

The original MLP training plots compare default SGD architecture experiments,
not the boosted-tree/blend estimators used by the root-level replay. They must
not be presented as learning curves for the final ensemble.
