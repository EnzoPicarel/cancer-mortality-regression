"""Explore nonlinear functions and target transforms around the Ridge baseline.

Uses only train.csv. The seed-42 15% local holdout is excluded from CV and
used once for checking the CV-selected candidate.
Run: python advanced_search.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler

from model_search import make_features, make_pipeline
from train_models import ID, RANDOM_SEED, TARGET, correct_median_age
from ridge_tuning import variant_features


HERE = Path(__file__).resolve().parent.parent
DEV_SPLIT_SEED = 2026
SPLINE_FIELDS = [
    "incidenceRate", "medIncome", "povertyPercent", "PctPublicCoverageAlone",
    "PctBachDeg25_Over", "PctUnemployed16_Over", "PctHS25_Over",
    "PctPublicCoverage", "PctPrivateCoverage", "PctMarriedHouseholds",
    "PctEmployed16_Over", "PctHS18_24", "PctWhite", "PctBlack", "BirthRate",
]
INTERACTION_FIELDS = [
    "incidenceRate", "medIncome", "povertyPercent", "PctBachDeg25_Over",
    "PctHS25_Over", "PctUnemployed16_Over", "PctPublicCoverageAlone",
    "PctPrivateCoverage", "PctEmployed16_Over", "PctWhite", "PctBlack",
    "log1p_avgAnnCount", "log1p_avgDeathsPerYear", "log1p_popEst2015",
]


def gam_pipeline(X: pd.DataFrame, alpha: float) -> Pipeline:
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    spline = [c for c in SPLINE_FIELDS if c in nums]
    linear = [c for c in nums if c not in spline]
    transformers = []
    if spline:
        transformers.append((
            "spline",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("basis", SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False)),
                ("scale", StandardScaler()),
            ]),
            spline,
        ))
    if linear:
        transformers.append((
            "linear_numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            linear,
        ))
    if cats:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cats,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", Ridge(alpha=alpha))])


def interaction_pipeline(X: pd.DataFrame, alpha: float) -> Pipeline:
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    core = [c for c in INTERACTION_FIELDS if c in nums]
    rest = [c for c in nums if c not in core]
    transformers = []
    if core:
        transformers.append((
            "interactions",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ]),
            core,
        ))
    if rest:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            rest,
        ))
    if cats:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cats,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", Ridge(alpha=alpha))])


def candidates(X: pd.DataFrame) -> dict[str, object]:
    options = {
        "Ridge baseline alpha=1": make_pipeline(X, Ridge(alpha=1.0)),
        "Ridge baseline alpha=.3": make_pipeline(X, Ridge(alpha=0.3)),
        "Spline Ridge alpha=.3": gam_pipeline(X, 0.3),
        "Spline Ridge alpha=1": gam_pipeline(X, 1.0),
        "Spline Ridge alpha=3": gam_pipeline(X, 3.0),
        "Interaction Ridge alpha=.3": interaction_pipeline(X, 0.3),
        "Interaction Ridge alpha=1": interaction_pipeline(X, 1.0),
        "Interaction Ridge alpha=3": interaction_pipeline(X, 3.0),
    }
    base_pipe = make_pipeline(X, Ridge(alpha=1.0))
    options["Log-target Ridge alpha=1"] = TransformedTargetRegressor(
        regressor=base_pipe, func=np.log, inverse_func=np.exp
    )
    # Tree model candidates use the same state, log and imputation features.
    options["HistGB leaf=15, l2=0.1"] = make_pipeline(
        X,
        HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=10, l2_regularization=0.1, random_state=RANDOM_SEED,
        ),
    )
    options["HistGB leaf=31, l2=1"] = make_pipeline(
        X,
        HistGradientBoostingRegressor(
            learning_rate=0.04, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=15, l2_regularization=1.0, random_state=RANDOM_SEED,
        ),
    )
    return options


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, changed = correct_median_age(raw)
    X = variant_features(data, "state + logs")
    y = data[TARGET].astype(float)

    row_ids = pd.Series(np.arange(len(data)))
    _, remainder = train_test_split(row_ids, test_size=0.30, random_state=RANDOM_SEED)
    _, holdout = train_test_split(remainder, test_size=0.50, random_state=RANDOM_SEED)
    holdout_ids = holdout.to_numpy()
    dev_ids = np.setdiff1d(np.arange(len(data)), holdout_ids)
    X_dev, y_dev = X.iloc[dev_ids], y.iloc[dev_ids]
    X_holdout, y_holdout = X.iloc[holdout_ids], y.iloc[holdout_ids]
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=DEV_SPLIT_SEED)
    rows = []

    print(f"MedianAge converted in {changed} training rows; development={len(dev_ids)}, holdout={len(holdout_ids)}.")
    for name, estimator in candidates(X_dev).items():
        scores = cross_validate(
            estimator, X_dev, y_dev, cv=cv,
            scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
            n_jobs=1,
        )
        row = {
            "model": name,
            "cv_R2_mean": scores["test_r2"].mean(),
            "cv_R2_std": scores["test_r2"].std(ddof=1),
            "cv_MSE_mean": -scores["test_mse"].mean(),
            "cv_MSE_std": scores["test_mse"].std(ddof=1),
        }
        rows.append(row)
        print(f"{name:31s} CV R2={row['cv_R2_mean']:.4f} +/- {row['cv_R2_std']:.4f}; MSE={row['cv_MSE_mean']:.2f}")

    results = pd.DataFrame(rows).sort_values("cv_R2_mean", ascending=False)
    results.to_csv(HERE / "results" / "advanced_cv_search.csv", index=False)
    best_name = str(results.iloc[0]["model"])
    best_estimator = candidates(X_dev)[best_name]
    best_estimator.fit(X_dev, y_dev)
    pred = best_estimator.predict(X_holdout)
    holdout_r2 = r2_score(y_holdout, pred)
    holdout_mse = mean_squared_error(y_holdout, pred)
    print("\nRanked CV results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nCV-selected {best_name}; local holdout R2={holdout_r2:.4f}, MSE={holdout_mse:.2f}.")
    pd.DataFrame([{
        "model": best_name,
        "cv_R2_mean": results.iloc[0].cv_R2_mean,
        "cv_R2_std": results.iloc[0].cv_R2_std,
        "cv_MSE_mean": results.iloc[0].cv_MSE_mean,
        "holdout_R2": holdout_r2,
        "holdout_MSE": holdout_mse,
    }]).to_csv(HERE / "results" / "advanced_best_holdout.csv", index=False)

    if holdout_r2 > 0.87845:  # at least +0.02 over the previous .85845 holdout
        final_model = candidates(X)[best_name]
        final_model.fit(X, y)
        kaggle_test, test_changed = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
        X_test = variant_features(kaggle_test, "state + logs")
        submission = pd.DataFrame({ID: kaggle_test[ID], TARGET: final_model.predict(X_test)})
        submission.to_csv(HERE / "results" / "submission_advanced.csv", index=False)
        joblib.dump(
            {"model": final_model, "model_name": best_name,
             "median_age_threshold": 100, "median_age_divisor": 12,
             "train_rows_corrected": changed, "test_rows_corrected": test_changed,
             "cv_R2_mean": float(results.iloc[0].cv_R2_mean),
             "cv_R2_std": float(results.iloc[0].cv_R2_std),
             "holdout_R2": holdout_r2},
            HERE / "models" / "advanced_best_model.joblib",
        )
        print(f"\nMeaningful holdout gain; wrote submission_advanced.csv ({len(submission)} rows).")
    else:
        print("\nNo +0.02 local holdout gain over the Ridge candidate; keep submission_improved.csv.")


if __name__ == "__main__":
    main()
