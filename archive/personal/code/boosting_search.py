"""Compare CatBoost and LightGBM with the current blend on repeated CV.

All experiments use only personal_kaggle/data/train.csv. They write only to
personal_kaggle/results and personal_kaggle/models.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

HERE = Path(__file__).resolve().parent.parent

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ensemble_search import tree_pipeline
from feature_search import engineered
from model_search import make_pipeline
from train_models import TARGET, correct_median_age

SEED = 9351


def make_candidates(X):
    candidates = {
        "Current blend 35% Ridge + 65% HistGB": "blend",
        "Ridge alpha=.75": make_pipeline(X, Ridge(alpha=0.75)),
        "HistGB leaves15": tree_pipeline(
            X,
            HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            ),
        ),
    }
    for depth, iterations, rate, l2 in [
        (4, 900, 0.035, 5),
        (5, 900, 0.035, 5),
        (6, 1000, 0.03, 7),
        (7, 1000, 0.03, 8),
    ]:
        name = f"CatBoost depth={depth} iter={iterations} lr={rate} l2={l2}"
        candidates[name] = {
            "type": "catboost", "depth": depth, "iterations": iterations,
            "learning_rate": rate, "l2_leaf_reg": l2,
        }
    for leaves, iterations, rate, min_child, reg in [
        (7, 700, 0.03, 20, 5),
        (15, 700, 0.03, 20, 5),
        (31, 700, 0.03, 20, 5),
        (15, 1100, 0.02, 25, 8),
    ]:
        name = f"LightGBM leaves={leaves} iter={iterations} lr={rate} child={min_child} l2={reg}"
        candidates[name] = {
            "type": "lightgbm", "num_leaves": leaves, "n_estimators": iterations,
            "learning_rate": rate, "min_child_samples": min_child,
            "reg_lambda": reg,
        }
    return candidates


def fit_candidate(spec, X_train, y_train, X_valid):
    if spec == "blend":
        ridge = make_pipeline(X_train, Ridge(alpha=0.75)).fit(X_train, y_train)
        hist = tree_pipeline(
            X_train,
            HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            ),
        ).fit(X_train, y_train)
        return 0.35 * ridge.predict(X_valid) + 0.65 * hist.predict(X_valid)
    if isinstance(spec, dict) and spec.get("type") == "catboost":
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        train = X_train.copy()
        valid = X_valid.copy()
        for col in cat_cols:
            train[col] = train[col].fillna("Unknown").astype(str)
            valid[col] = valid[col].fillna("Unknown").astype(str)
        params = {key: value for key, value in spec.items() if key != "type"}
        model = CatBoostRegressor(
            loss_function="RMSE", random_seed=SEED, verbose=False,
            allow_writing_files=False, thread_count=4, **params,
        )
        model.fit(train, y_train, cat_features=cat_cols)
        return model.predict(valid)
    if isinstance(spec, dict) and spec.get("type") == "lightgbm":
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        train = X_train.copy()
        valid = X_valid.copy()
        for col in cat_cols:
            categories = pd.Index(pd.concat([train[col], valid[col]], ignore_index=True).fillna("Unknown").astype(str).unique())
            train[col] = pd.Categorical(train[col].fillna("Unknown").astype(str), categories=categories)
            valid[col] = pd.Categorical(valid[col].fillna("Unknown").astype(str), categories=categories)
        params = {key: value for key, value in spec.items() if key != "type"}
        model = LGBMRegressor(
            objective="regression", random_state=SEED, verbosity=-1,
            n_jobs=4, deterministic=True, force_col_wise=True, **params,
        )
        model.fit(train, y_train, categorical_feature=cat_cols)
        return model.predict(valid)
    model = clone(spec).fit(X_train, y_train)
    return model.predict(X_valid)


def main():
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(raw)
    y = data[TARGET].to_numpy(dtype=float)
    X = engineered(data, "ratios")
    candidates = make_candidates(X)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    predictions = {name: [] for name in candidates}
    fold_scores = {name: [] for name in candidates}
    splits = list(cv.split(X))

    for fold, (tr, va) in enumerate(splits, start=1):
        for name, spec in candidates.items():
            pred = fit_candidate(spec, X.iloc[tr], y[tr], X.iloc[va])
            predictions[name].append((va, pred))
            fold_scores[name].append(r2_score(y[va], pred))
        if fold % 5 == 0:
            print(f"Completed {fold}/{len(splits)} folds", flush=True)

    rows = []
    for name in candidates:
        repeat_r2, repeat_mse = [], []
        for repeat in range(3):
            actual, predicted = [], []
            for fold in range(repeat * 5, (repeat + 1) * 5):
                idx, pred = predictions[name][fold]
                actual.extend(y[idx])
                predicted.extend(pred)
            repeat_r2.append(r2_score(actual, predicted))
            repeat_mse.append(mean_squared_error(actual, predicted))
        rows.append({
            "candidate": name,
            "mean_fold_r2": float(np.mean(fold_scores[name])),
            "std_fold_r2": float(np.std(fold_scores[name], ddof=1)),
            "mean_repeat_r2": float(np.mean(repeat_r2)),
            "std_repeat_r2": float(np.std(repeat_r2, ddof=1)),
            "mean_repeat_mse": float(np.mean(repeat_mse)),
        })

    results = pd.DataFrame(rows).sort_values("mean_repeat_r2", ascending=False)
    out = HERE / "results" / "boosting_cv_comparison.csv"
    results.to_csv(out, index=False)
    print("\nRepeated 5-fold CV results (3 repeats):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
