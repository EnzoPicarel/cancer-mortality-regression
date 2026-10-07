"""Tune CatBoost with inner-fold early stopping and compare blend weights."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split

HERE = Path(__file__).resolve().parent.parent

from catboost import CatBoostRegressor

from ensemble_search import tree_pipeline
from feature_search import engineered
from model_search import make_pipeline
from preprocessing_search import make_extended_features
from train_models import TARGET, correct_median_age

SEED = 9351
MAX_ITERATIONS = 2500
EARLY_STOPPING_ROUNDS = 120

CONFIGS = {
    "depth4_lr.03_l2=5": {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    "depth5_lr.03_l2=7": {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 7},
    "depth6_lr.025_l2=10": {"depth": 6, "learning_rate": 0.025, "l2_leaf_reg": 10},
}


def cat_model(config, iterations):
    return CatBoostRegressor(
        loss_function="RMSE", iterations=iterations, random_seed=SEED,
        verbose=False, allow_writing_files=False, thread_count=4,
        **config,
    )


def prep_cat(X):
    out = X.copy()
    cats = out.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cats:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out, cats


def repeat_metrics(fold_predictions, y, weight=1.0, base_predictions=None):
    scores, mses = [], []
    for repeat in range(3):
        actual, pred_all = [], []
        for fold in range(repeat * 5, (repeat + 1) * 5):
            idx, pred = fold_predictions[fold]
            if base_predictions is not None:
                base_idx, base_pred = base_predictions[fold]
                assert np.array_equal(idx, base_idx)
                pred = weight * pred + (1 - weight) * base_pred
            actual.extend(y[idx])
            pred_all.extend(pred)
        scores.append(r2_score(actual, pred_all))
        mses.append(mean_squared_error(actual, pred_all))
    return scores, mses


def main():
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(raw)
    y = data[TARGET].to_numpy(dtype=float)
    X_base = engineered(data, "ratios")
    X_extended, cat_cols = prep_cat(make_extended_features(data))
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    splits = list(cv.split(X_base))

    base_oof = []
    for fold, (tr, va) in enumerate(splits, start=1):
        ridge = make_pipeline(X_base, Ridge(alpha=0.75)).fit(X_base.iloc[tr], y[tr])
        hist = tree_pipeline(
            X_base,
            HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            ),
        ).fit(X_base.iloc[tr], y[tr])
        pred = 0.35 * ridge.predict(X_base.iloc[va]) + 0.65 * hist.predict(X_base.iloc[va])
        base_oof.append((va, pred))
        if fold % 5 == 0:
            print(f"Built current-blend OOF predictions: {fold}/{len(splits)} folds", flush=True)

    cat_oof = {name: [] for name in CONFIGS}
    best_iterations = {name: [] for name in CONFIGS}
    for fold, (tr, va) in enumerate(splits, start=1):
        inner_train, inner_stop = train_test_split(
            tr, test_size=0.15, random_state=SEED + fold,
        )
        for name, config in CONFIGS.items():
            early_model = cat_model(config, MAX_ITERATIONS)
            early_model.fit(
                X_extended.iloc[inner_train], y[inner_train],
                cat_features=cat_cols,
                eval_set=(X_extended.iloc[inner_stop], y[inner_stop]),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            iterations = max(1, int(early_model.best_iteration_) + 1)
            best_iterations[name].append(iterations)
            refit = cat_model(config, iterations)
            refit.fit(X_extended.iloc[tr], y[tr], cat_features=cat_cols)
            pred = refit.predict(X_extended.iloc[va])
            cat_oof[name].append((va, pred))
        if fold % 5 == 0:
            print(f"Completed early-stopped CatBoost: {fold}/{len(splits)} folds", flush=True)

    rows = []
    base_scores, base_mses = repeat_metrics(base_oof, y)
    rows.append({
        "candidate": "Current blend 35% Ridge + 65% HistGB",
        "mean_repeat_r2": float(np.mean(base_scores)),
        "std_repeat_r2": float(np.std(base_scores, ddof=1)),
        "mean_repeat_mse": float(np.mean(base_mses)),
        "blend_catboost_weight": 0.0,
        "median_best_iteration": np.nan,
    })
    for name, folds in cat_oof.items():
        scores, mses = repeat_metrics(folds, y)
        rows.append({
            "candidate": f"CatBoost {name}",
            "mean_repeat_r2": float(np.mean(scores)),
            "std_repeat_r2": float(np.std(scores, ddof=1)),
            "mean_repeat_mse": float(np.mean(mses)),
            "blend_catboost_weight": 1.0,
            "median_best_iteration": float(np.median(best_iterations[name])),
        })
        for cat_weight in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
            scores, mses = repeat_metrics(folds, y, weight=cat_weight, base_predictions=base_oof)
            rows.append({
                "candidate": f"Current + CatBoost {name}; cat_weight={cat_weight:.2f}",
                "mean_repeat_r2": float(np.mean(scores)),
                "std_repeat_r2": float(np.std(scores, ddof=1)),
                "mean_repeat_mse": float(np.mean(mses)),
                "blend_catboost_weight": cat_weight,
                "median_best_iteration": float(np.median(best_iterations[name])),
            })

    results = pd.DataFrame(rows).sort_values("mean_repeat_r2", ascending=False)
    out = HERE / "results" / "catboost_refinement_cv.csv"
    results.to_csv(out, index=False)
    print("\nTop CV candidates:")
    print(results.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
