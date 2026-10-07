"""Final focused tuning around the winning Ridge + HistGB blend.

All tuning and scoring uses only train.csv labels. OOF predictions are kept
aligned so blend weights are selected on identical repeated CV partitions.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from feature_search import engineered
from model_search import make_pipeline
from train_models import ID, TARGET, correct_median_age
from ensemble_search import tree_pipeline

HERE = Path(__file__).resolve().parent.parent
SEED = 9351


def main():
    raw = pd.read_csv(HERE / "data" / "train.csv")
    train, _ = correct_median_age(raw)
    y = train[TARGET].to_numpy(dtype=float)
    X = engineered(train, "ratios")

    configs = {
        "HistGB L10 lr.04 it350 leaf18 l2=3": dict(max_leaf_nodes=10, learning_rate=.04, max_iter=350, min_samples_leaf=18, l2_regularization=3),
        "HistGB L15 lr.04 it350 leaf18 l2=3 (current)": dict(max_leaf_nodes=15, learning_rate=.04, max_iter=350, min_samples_leaf=18, l2_regularization=3),
        "HistGB L20 lr.04 it350 leaf18 l2=3": dict(max_leaf_nodes=20, learning_rate=.04, max_iter=350, min_samples_leaf=18, l2_regularization=3),
        "HistGB L25 lr.04 it350 leaf18 l2=3": dict(max_leaf_nodes=25, learning_rate=.04, max_iter=350, min_samples_leaf=18, l2_regularization=3),
        "HistGB L15 lr.03 it500 leaf18 l2=3": dict(max_leaf_nodes=15, learning_rate=.03, max_iter=500, min_samples_leaf=18, l2_regularization=3),
        "HistGB L15 lr.05 it250 leaf18 l2=3": dict(max_leaf_nodes=15, learning_rate=.05, max_iter=250, min_samples_leaf=18, l2_regularization=3),
        "HistGB L15 lr.04 it350 leaf12 l2=3": dict(max_leaf_nodes=15, learning_rate=.04, max_iter=350, min_samples_leaf=12, l2_regularization=3),
        "HistGB L15 lr.04 it350 leaf30 l2=3": dict(max_leaf_nodes=15, learning_rate=.04, max_iter=350, min_samples_leaf=30, l2_regularization=3),
        "HistGB L20 lr.05 it350 leaf25 l2=5": dict(max_leaf_nodes=20, learning_rate=.05, max_iter=350, min_samples_leaf=25, l2_regularization=5),
    }
    estimators = {name: tree_pipeline(X, HistGradientBoostingRegressor(**params, random_state=SEED))
                  for name, params in configs.items()}
    for alpha in [.3, .75, 1.5]:
        name = f"Ridge alpha={alpha}"
        estimators[name] = make_pipeline(X, Ridge(alpha=alpha))

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof = {name: [] for name in estimators}
    fold_scores = {name: [] for name in estimators}
    for fold, (tr, va) in enumerate(cv.split(X), start=1):
        for name, base in estimators.items():
            model = clone(base).fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            oof[name].append((va, pred))
            fold_scores[name].append(r2_score(y[va], pred))
        if fold % 5 == 0:
            print(f"Completed {fold}/15 folds", flush=True)

    rows = []
    for name in estimators:
        fold_r2 = np.asarray(fold_scores[name])
        rows.append({"candidate": name, "cv_r2_mean": fold_r2.mean(), "cv_r2_std_fold": fold_r2.std(ddof=1), "cv_mse_mean": np.mean([mean_squared_error(y[idx], p) for idx, p in oof[name]])})

    ridge_names = [f"Ridge alpha={a}" for a in [.3, .75, 1.5]]
    hgb_names = list(configs)
    blend_rows = []
    for ridge_name in ridge_names:
        for hgb_name in hgb_names:
            for ridge_weight in np.round(np.arange(.20, .601, .05), 2):
                repeat_r2, repeat_mse = [], []
                for rep in range(3):
                    actual, predicted = [], []
                    for f in range(rep * 5, (rep + 1) * 5):
                        idx, ridge_pred = oof[ridge_name][f]
                        idx_h, hgb_pred = oof[hgb_name][f]
                        assert np.array_equal(idx, idx_h)
                        actual.extend(y[idx])
                        predicted.extend(ridge_weight * ridge_pred + (1 - ridge_weight) * hgb_pred)
                    repeat_r2.append(r2_score(actual, predicted))
                    repeat_mse.append(mean_squared_error(actual, predicted))
                blend_rows.append({
                    "candidate": f"Blend {ridge_name} + {hgb_name}",
                    "ridge_weight": ridge_weight,
                    "cv_r2_mean": np.mean(repeat_r2),
                    "cv_r2_std_repeat": np.std(repeat_r2, ddof=1),
                    "cv_mse_mean": np.mean(repeat_mse),
                })
    blends = pd.DataFrame(blend_rows).sort_values("cv_r2_mean", ascending=False)
    pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).to_csv(HERE / "results" / "final_tuning_individual.csv", index=False)
    blends.to_csv(HERE / "results" / "final_tuning_blends.csv", index=False)
    print("\nTop individual models:")
    print(pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTop blends:")
    print(blends.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Refit the best blend using every labeled row and create its Kaggle file.
    best = blends.iloc[0]
    ridge_name = next(n for n in ridge_names if n in best.candidate)
    hgb_name = str(best.candidate).split(" + ", 1)[1]
    weight = float(best.ridge_weight)
    ridge_model = clone(estimators[ridge_name]).fit(X, y)
    hgb_model = clone(estimators[hgb_name]).fit(X, y)
    test, _ = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
    Xt = engineered(test, "ratios")
    pred = weight * ridge_model.predict(Xt) + (1 - weight) * hgb_model.predict(Xt)
    submission = pd.DataFrame({ID: test[ID], TARGET: pred})
    submission.to_csv(HERE / "deliverables" / "submission.csv", index=False)
    joblib.dump({"ridge": ridge_model, "hist_gradient_boosting": hgb_model,
                 "ridge_weight": weight, "ridge_config": ridge_name,
                 "histgb_config": hgb_name, "cv_r2_mean": float(best.cv_r2_mean),
                 "cv_r2_std_repeat": float(best.cv_r2_std_repeat)}, HERE / "models" / "final_tuned_model.joblib")
    print(f"\nSelected {best.candidate}, Ridge weight {weight:.2f}; CV R2={best.cv_r2_mean:.5f} +/- {best.cv_r2_std_repeat:.5f}.")
    print(f"Wrote submission.csv; prediction range {pred.min():.2f}–{pred.max():.2f}.")


if __name__ == "__main__":
    main()
