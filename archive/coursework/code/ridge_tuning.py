"""Tune a regularized regression using only train.csv labels.

The seed-42 15% local test split remains outside hyperparameter selection.
Run: python ridge_tuning.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_validate

from model_search import make_features, make_pipeline
from train_models import ID, RANDOM_SEED, TARGET, correct_median_age


HERE = Path(__file__).resolve().parent.parent
ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
VARIANTS = [
    "state + logs",
    "state, no logs",
    "no state + logs",
    "state + logs, no MedianAge",
    "state + logs, no population/count",
]


def variant_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    add_state = not variant.startswith("no state")
    X = make_features(frame, add_state=add_state)
    if "no logs" in variant:
        X = X.drop(columns=[c for c in X.columns if c.startswith("log1p_")])
    if "no MedianAge" in variant:
        X = X.drop(columns=["MedianAge"], errors="ignore")
    if "no population/count" in variant:
        columns = [
            "avgAnnCount", "avgDeathsPerYear", "popEst2015", "studyPerCap",
            "log1p_avgAnnCount", "log1p_avgDeathsPerYear",
            "log1p_popEst2015", "log1p_studyPerCap",
        ]
        X = X.drop(columns=columns, errors="ignore")
    return X


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, converted = correct_median_age(raw)
    y = data[TARGET].astype(float)

    # Match the established 70/15/15 split: 85% development data and a
    # completely withheld 15% local test used only once after CV selection.
    row_ids = pd.Series(np.arange(len(data)))
    train_ids, remainder_ids = train_test_split(
        row_ids, test_size=0.30, random_state=RANDOM_SEED
    )
    valid_ids, holdout_ids = train_test_split(
        remainder_ids, test_size=0.50, random_state=RANDOM_SEED
    )
    dev_ids = pd.concat([train_ids, valid_ids], ignore_index=True).to_numpy()
    holdout_ids = holdout_ids.to_numpy()
    y_dev, y_holdout = y.iloc[dev_ids], y.iloc[holdout_ids]
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2026)
    rows = []

    print(
        f"Corrected MedianAge in {converted} training rows; "
        f"development={len(dev_ids)}, untouched local test={len(holdout_ids)}."
    )
    for variant in VARIANTS:
        X = variant_features(data, variant)
        X_dev = X.iloc[dev_ids]
        for alpha in ALPHAS:
            estimator = Ridge(alpha=alpha)
            pipe = make_pipeline(X_dev, estimator)
            scores = cross_validate(
                pipe,
                X_dev,
                y_dev,
                cv=cv,
                scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                n_jobs=1,
            )
            rows.append(
                {
                    "feature_variant": variant,
                    "alpha": alpha,
                    "cv_R2_mean": scores["test_r2"].mean(),
                    "cv_R2_std": scores["test_r2"].std(ddof=1),
                    "cv_MSE_mean": -scores["test_mse"].mean(),
                    "cv_MSE_std": scores["test_mse"].std(ddof=1),
                }
            )
        print(f"Finished CV grid: {variant}")

    cv_results = pd.DataFrame(rows).sort_values("cv_R2_mean", ascending=False)
    cv_results.to_csv(HERE / "results" / "ridge_cv_search.csv", index=False)
    print("\nTop CV configurations:")
    print(cv_results.head(12).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Evaluate the single CV-selected configuration on the untouched local
    # test rows, then refit that configuration on all labeled rows for Kaggle.
    best = cv_results.iloc[0]
    best_variant, best_alpha = str(best.feature_variant), float(best.alpha)
    X_all = variant_features(data, best_variant)
    X_dev, X_holdout = X_all.iloc[dev_ids], X_all.iloc[holdout_ids]
    holdout_pipe = make_pipeline(X_dev, Ridge(alpha=best_alpha))
    holdout_pipe.fit(X_dev, y_dev)
    holdout_pred = holdout_pipe.predict(X_holdout)
    holdout_r2 = r2_score(y_holdout, holdout_pred)
    holdout_mse = mean_squared_error(y_holdout, holdout_pred)
    baseline_mse = mean_squared_error(y_holdout, np.full(len(y_holdout), y_dev.mean()))
    print(
        f"\nSelected by repeated CV: {best_variant}, Ridge alpha={best_alpha:g}; "
        f"CV R2={best.cv_R2_mean:.4f} +/- {best.cv_R2_std:.4f}."
    )
    print(
        f"Untouched local test: R2={holdout_r2:.4f}, MSE={holdout_mse:.2f}; "
        f"mean baseline MSE={baseline_mse:.2f}."
    )
    pd.DataFrame(
        [{
            "feature_variant": best_variant,
            "alpha": best_alpha,
            "cv_R2_mean": best.cv_R2_mean,
            "cv_R2_std": best.cv_R2_std,
            "cv_MSE_mean": best.cv_MSE_mean,
            "holdout_R2": holdout_r2,
            "holdout_MSE": holdout_mse,
            "holdout_mean_baseline_MSE": baseline_mse,
        }]
    ).to_csv(HERE / "results" / "ridge_best_holdout.csv", index=False)

    final_pipe = make_pipeline(X_all, Ridge(alpha=best_alpha))
    final_pipe.fit(X_all, y)
    kaggle_test, test_converted = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
    X_kaggle = variant_features(kaggle_test, best_variant)
    prediction = final_pipe.predict(X_kaggle)
    submission = pd.DataFrame({ID: kaggle_test[ID], TARGET: prediction})
    submission.to_csv(HERE / "results" / "submission_improved.csv", index=False)
    joblib.dump(
        {
            "pipeline": final_pipe,
            "model_name": "Ridge",
            "alpha": best_alpha,
            "feature_variant": best_variant,
            "median_age_threshold": 100,
            "median_age_divisor": 12,
            "median_age_rows_converted_train": converted,
            "median_age_rows_converted_test": test_converted,
            "cv_R2_mean": float(best.cv_R2_mean),
            "cv_R2_std": float(best.cv_R2_std),
            "holdout_R2": holdout_r2,
            "random_seed": RANDOM_SEED,
        },
        HERE / "models" / "ridge_best_model.joblib",
    )
    print(
        f"Created submission_improved.csv ({len(submission)} rows; "
        f"prediction range {prediction.min():.2f} to {prediction.max():.2f}) and ridge_best_model.joblib."
    )
    print("This submission uses no Kaggle test labels; upload only after reviewing the holdout result.")


if __name__ == "__main__":
    main()
