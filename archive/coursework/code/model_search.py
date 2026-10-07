"""Cross-validated search for stronger tabular baselines.

This uses only train.csv labels. It never reads or joins hidden Kaggle targets.
Run: python model_search.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from train_models import TARGET, correct_median_age


HERE = Path(__file__).resolve().parent.parent
SEED = 42


def make_features(frame: pd.DataFrame, add_state: bool = True) -> pd.DataFrame:
    data = frame.drop(columns=[TARGET, "id", "Geography"], errors="ignore").copy()
    if add_state:
        # County names are unique. State is a repeated, lower-cardinality
        # regional feature extracted from the supplied county label.
        data["state"] = frame["Geography"].astype(str).str.rsplit(",", n=1).str[-1].str.strip()
    for column in ["avgAnnCount", "avgDeathsPerYear", "popEst2015", "studyPerCap"]:
        if column in data:
            data[f"log1p_{column}"] = np.log1p(data[column].clip(lower=0))
    return data


def make_pipeline(X: pd.DataFrame, model) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True)),
         ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def model_candidates() -> dict[str, object]:
    return {
        "Linear": LinearRegression(),
        "Ridge a=1": Ridge(alpha=1.0),
        "Ridge a=10": Ridge(alpha=10.0),
        "Ridge a=50": Ridge(alpha=50.0),
        "ElasticNet a=.01 l1=.2": ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000),
        "SVR RBF C=10": SVR(C=10, epsilon=3.0, gamma="scale"),
        "SVR RBF C=100": SVR(C=100, epsilon=3.0, gamma="scale"),
        "KNN k=15": KNeighborsRegressor(n_neighbors=15, weights="distance", p=1),
        "RandomForest leaf=2": RandomForestRegressor(
            n_estimators=350, max_features=0.8, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
        "RandomForest leaf=5": RandomForestRegressor(
            n_estimators=350, max_features=0.8, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees leaf=2": ExtraTreesRegressor(
            n_estimators=350, max_features=0.8, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees leaf=5": ExtraTreesRegressor(
            n_estimators=350, max_features=0.8, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
        "HistGB leaf=7": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=7,
            l2_regularization=2, random_state=SEED,
        ),
        "HistGB leaf=15": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=2, random_state=SEED,
        ),
        "GradientBoosting depth=2": GradientBoostingRegressor(
            n_estimators=350, learning_rate=0.03, max_depth=2,
            min_samples_leaf=5, loss="huber", random_state=SEED,
        ),
    }


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    corrected, changed = correct_median_age(raw)
    X = make_features(corrected, add_state=True)
    y = corrected[TARGET].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []

    print(f"5-fold CV, seed={SEED}; {len(y)} labeled rows; MedianAge corrected in {changed} rows.")
    print(f"Feature matrix before one-hot: {X.shape[1]} columns; state feature enabled.")
    for name, estimator in model_candidates().items():
        pipe = make_pipeline(X, estimator)
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
            n_jobs=1,
            return_train_score=True,
        )
        row = {
            "model": name,
            "cv_R2_mean": scores["test_r2"].mean(),
            "cv_R2_std": scores["test_r2"].std(ddof=1),
            "cv_MSE_mean": -scores["test_mse"].mean(),
            "cv_MSE_std": scores["test_mse"].std(ddof=1),
            "train_R2_mean": scores["train_r2"].mean(),
            "train_R2_std": scores["train_r2"].std(ddof=1),
        }
        rows.append(row)
        print(
            f"{name:27s} CV R2={row['cv_R2_mean']:.4f} +/- {row['cv_R2_std']:.4f}; "
            f"MSE={row['cv_MSE_mean']:.2f}"
        )

    results = pd.DataFrame(rows).sort_values("cv_R2_mean", ascending=False)
    results.to_csv(HERE / "results" / "cv_model_search.csv", index=False)
    print("\nRanked results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nWrote cv_model_search.csv")


if __name__ == "__main__":
    main()
