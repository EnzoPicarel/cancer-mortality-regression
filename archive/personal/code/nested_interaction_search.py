"""Leakage-aware interaction search with feature selection inside CV folds."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from model_search import make_features, make_pipeline
from train_models import ID, RANDOM_SEED, TARGET, correct_median_age
from ridge_tuning import variant_features


HERE = Path(__file__).resolve().parent.parent
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
K_VALUES = [8, 12, 16, 24, "all"]


def nested_poly_pipeline(X: pd.DataFrame, alpha: float, k) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    if k != "all":
        numeric_steps.append(("select", SelectKBest(score_func=f_regression, k=int(k))))
    numeric_steps.append((
        "pairwise_interactions",
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    ))
    transformers = [
        ("numeric_interactions", Pipeline(numeric_steps), numeric),
    ]
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", Ridge(alpha=alpha))])


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, converted = correct_median_age(raw)
    X = variant_features(data, "state + logs")
    y = data[TARGET].astype(float)
    row_ids = pd.Series(np.arange(len(data)))
    _, remainder = train_test_split(row_ids, test_size=0.30, random_state=RANDOM_SEED)
    _, holdout = train_test_split(remainder, test_size=0.50, random_state=RANDOM_SEED)
    holdout_ids = holdout.to_numpy()
    dev_ids = np.setdiff1d(np.arange(len(data)), holdout_ids)
    X_dev, y_dev = X.iloc[dev_ids], y.iloc[dev_ids]
    X_holdout, y_holdout = X.iloc[holdout_ids], y.iloc[holdout_ids]
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2026)
    rows = []

    print(f"Corrected MedianAge in {converted} rows; CV development={len(dev_ids)}; local holdout={len(holdout_ids)}.")
    for k in K_VALUES:
        for alpha in ALPHAS:
            name = f"nested top-{k} numeric + pairwise interactions; Ridge alpha={alpha:g}"
            estimator = nested_poly_pipeline(X_dev, alpha, k)
            scores = cross_validate(
                estimator, X_dev, y_dev, cv=cv,
                scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
                n_jobs=1,
            )
            rows.append({
                "model": name,
                "k_numeric": k,
                "alpha": alpha,
                "cv_R2_mean": scores["test_r2"].mean(),
                "cv_R2_std": scores["test_r2"].std(ddof=1),
                "cv_MSE_mean": -scores["test_mse"].mean(),
                "cv_MSE_std": scores["test_mse"].std(ddof=1),
            })
        print(f"Finished nested feature-selection grid for k={k}.")

    results = pd.DataFrame(rows).sort_values("cv_R2_mean", ascending=False)
    results.to_csv(HERE / "results" / "nested_interaction_cv.csv", index=False)
    best = results.iloc[0]
    best_k = best.k_numeric
    if best_k != "all":
        best_k = int(best_k)
    best_alpha = float(best.alpha)
    selected = nested_poly_pipeline(X_dev, best_alpha, best_k)
    selected.fit(X_dev, y_dev)
    pred = selected.predict(X_holdout)
    holdout_r2 = r2_score(y_holdout, pred)
    holdout_mse = mean_squared_error(y_holdout, pred)
    print("\nTop nested-CV configurations:")
    print(results.head(12).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nCV-selected k={best_k}, alpha={best_alpha:g}; CV R2={best.cv_R2_mean:.4f} +/- {best.cv_R2_std:.4f}; local holdout R2={holdout_r2:.4f}, MSE={holdout_mse:.2f}.")

    pd.DataFrame([{
        "k_numeric": best_k,
        "alpha": best_alpha,
        "cv_R2_mean": best.cv_R2_mean,
        "cv_R2_std": best.cv_R2_std,
        "cv_MSE_mean": best.cv_MSE_mean,
        "holdout_R2": holdout_r2,
        "holdout_MSE": holdout_mse,
    }]).to_csv(HERE / "results" / "nested_interaction_holdout.csv", index=False)

    full_model = nested_poly_pipeline(X, best_alpha, best_k)
    full_model.fit(X, y)
    kaggle_test, test_converted = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
    X_test = variant_features(kaggle_test, "state + logs")
    prediction = full_model.predict(X_test)
    submission = pd.DataFrame({ID: kaggle_test[ID], TARGET: prediction})
    submission.to_csv(HERE / "results" / "submission_nested.csv", index=False)
    joblib.dump({
        "pipeline": full_model,
        "k_numeric": best_k,
        "alpha": best_alpha,
        "median_age_threshold": 100,
        "median_age_divisor": 12,
        "train_rows_corrected": converted,
        "test_rows_corrected": test_converted,
        "cv_R2_mean": float(best.cv_R2_mean),
        "cv_R2_std": float(best.cv_R2_std),
        "holdout_R2": holdout_r2,
    }, HERE / "models" / "nested_interaction_model.joblib")
    print(f"Wrote submission_nested.csv ({len(submission)} predictions; range {prediction.min():.2f}–{prediction.max():.2f}).")


if __name__ == "__main__":
    main()
