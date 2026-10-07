"""Fit the best preprocessing-search blend and write a personal candidate."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent.parent

from catboost import CatBoostRegressor

from ensemble_search import tree_pipeline
from model_search import make_pipeline
from preprocessing_search import make_extended_features
from train_models import ID, TARGET, correct_median_age

SEED = 9351


def main():
    train_raw = pd.read_csv(HERE / "data" / "train.csv")
    test_raw = pd.read_csv(HERE / "data" / "test.csv")
    train, _ = correct_median_age(train_raw)
    test, _ = correct_median_age(test_raw)
    y = train[TARGET].to_numpy(dtype=float)
    X = make_extended_features(train)
    Xt = make_extended_features(test)

    ridge = make_pipeline(X, Ridge(alpha=0.75)).fit(X, y)
    hist = tree_pipeline(
        X,
        HistGradientBoostingRegressor(
            max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
            min_samples_leaf=12, l2_regularization=3, random_state=SEED,
        ),
    ).fit(X, y)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_cat, Xt_cat = X.copy(), Xt.copy()
    for col in cat_cols:
        X_cat[col] = X_cat[col].fillna("Unknown").astype(str)
        Xt_cat[col] = Xt_cat[col].fillna("Unknown").astype(str)
    cat = CatBoostRegressor(
        loss_function="RMSE", iterations=900, depth=5, learning_rate=0.035,
        l2_leaf_reg=5, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
    ).fit(X_cat, y, cat_features=cat_cols)

    p_ridge = ridge.predict(Xt)
    p_hist = hist.predict(Xt)
    p_cat = cat.predict(Xt_cat)
    p_current = 0.35 * p_ridge + 0.65 * p_hist
    predictions = 0.60 * p_current + 0.40 * p_cat

    submission = pd.DataFrame({ID: test[ID], TARGET: predictions})
    output = HERE / "submissions" / "submission_extended_blend.csv"
    submission.to_csv(output, index=False)
    joblib.dump({
        "ridge": ridge,
        "hist_gradient_boosting": hist,
        "catboost": cat,
        "weights": {"ridge": 0.21, "hist_gradient_boosting": 0.39, "catboost": 0.40},
        "cv_r2_mean_repeat": 0.90148,
        "baseline_cv_r2_mean_repeat": 0.89861,
        "note": "Personal experimental blend; CV-selected, not a Kaggle score.",
    }, HERE / "models" / "personal_extended_blend.joblib")
    print(f"Wrote {output} ({len(submission)} rows).")
    print(f"Prediction range: {predictions.min():.2f}–{predictions.max():.2f}")


if __name__ == "__main__":
    main()
