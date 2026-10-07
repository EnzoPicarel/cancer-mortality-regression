"""Fit the best exact-pipeline CatBoost refinement and write a Kaggle CSV."""
from pathlib import Path
import sys

import joblib
import numpy as np
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


def prep_cat(train, test):
    train, test = train.copy(), test.copy()
    cats = train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cats:
        train[col] = train[col].fillna("Unknown").astype(str)
        test[col] = test[col].fillna("Unknown").astype(str)
    return train, test, cats


def main():
    train_raw = pd.read_csv(HERE / "data" / "train.csv")
    test_raw = pd.read_csv(HERE / "data" / "test.csv")
    train, _ = correct_median_age(train_raw)
    test, _ = correct_median_age(test_raw)
    y = train[TARGET].to_numpy(dtype=float)
    X = make_extended_features(train)
    Xt = make_extended_features(test)
    X_cat, Xt_cat, cat_cols = prep_cat(X, Xt)

    ridge = make_pipeline(X, Ridge(alpha=0.75)).fit(X, y)
    hist = tree_pipeline(
        X,
        HistGradientBoostingRegressor(
            max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
            min_samples_leaf=12, l2_regularization=3, random_state=SEED,
        ),
    ).fit(X, y)
    original_cat = CatBoostRegressor(
        loss_function="RMSE", iterations=900, depth=5, learning_rate=0.035,
        l2_leaf_reg=5, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
    ).fit(X_cat, y, cat_features=cat_cols)
    long_cat = CatBoostRegressor(
        loss_function="RMSE", iterations=5000, depth=5, learning_rate=0.03,
        l2_leaf_reg=7, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
    ).fit(X_cat, y, cat_features=cat_cols)

    current_pred = (
        0.60 * (0.35 * ridge.predict(Xt) + 0.65 * hist.predict(Xt))
        + 0.40 * original_cat.predict(Xt_cat)
    )
    long_cat_pred = long_cat.predict(Xt_cat)
    predictions = 0.55 * current_pred + 0.45 * long_cat_pred

    submission = pd.DataFrame({ID: test[ID], TARGET: predictions})
    out = HERE / "submissions" / "submission_refined_catboost5000.csv"
    submission.to_csv(out, index=False)
    joblib.dump({
        "ridge": ridge,
        "hist_gradient_boosting": hist,
        "original_catboost": original_cat,
        "long_catboost": long_cat,
        "weights": {
            "current_submission_blend": 0.55,
            "long_catboost_5000_depth5_lr0.03_l2_7": 0.45,
            "current_internal_weights": {"ridge": 0.21, "hist_gradient_boosting": 0.39, "catboost_900": 0.40},
        },
        "cv_r2_mean_repeat": 0.90290,
        "cv_r2_std_repeat": 0.00099,
        "previous_submission_kaggle_r2": 0.91231,
        "note": "Personal Kaggle experiment. Validate on Kaggle before replacing the previous best.",
    }, HERE / "models" / "personal_refined_catboost5000.joblib")

    # Validate the generated file's schema, row count, ordering, and finite values.
    check = pd.read_csv(out)
    assert list(check.columns) == [ID, TARGET]
    assert len(check) == len(test_raw)
    assert np.array_equal(check[ID].to_numpy(), test_raw[ID].to_numpy())
    assert np.isfinite(check[TARGET].to_numpy()).all()
    print(f"Wrote {out} ({len(check)} rows).")
    print(f"Prediction range: {check[TARGET].min():.2f}–{check[TARGET].max():.2f}")
    print(f"IDs/order/schema/finite predictions verified; CV R2=0.90290 +/- 0.00099.")


if __name__ == "__main__":
    main()
