"""Compare longer CatBoost models against the exact 0.91231 submission pipeline."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

HERE = Path(__file__).resolve().parent.parent

from catboost import CatBoostRegressor

from ensemble_search import tree_pipeline
from model_search import make_pipeline
from preprocessing_search import make_extended_features
from train_models import ID, TARGET, correct_median_age

SEED = 9351
ITERATION_COUNTS = (2500, 3500, 5000)
CAT_WEIGHTS = (0.15, 0.25, 0.35, 0.45, 0.55)


def prep_cat(frame):
    out = frame.copy()
    cats = out.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cats:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out, cats


def fit_base_oof(X, y, splits):
    preds = []
    for fold, (tr, va) in enumerate(splits, start=1):
        ridge = make_pipeline(X, Ridge(alpha=0.75)).fit(X.iloc[tr], y[tr])
        hist = tree_pipeline(
            X,
            HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            ),
        ).fit(X.iloc[tr], y[tr])
        base = 0.35 * ridge.predict(X.iloc[va]) + 0.65 * hist.predict(X.iloc[va])

        cat = CatBoostRegressor(
            loss_function="RMSE", iterations=900, depth=5, learning_rate=0.035,
            l2_leaf_reg=5, random_seed=SEED, verbose=False,
            allow_writing_files=False, thread_count=4,
        )
        cat.fit(X.iloc[tr], y[tr], cat_features=X.attrs["cat_cols"])
        cat_pred = cat.predict(X.iloc[va])
        preds.append((va, 0.60 * base + 0.40 * cat_pred))
        if fold % 5 == 0:
            print(f"Exact current submission OOF: {fold}/{len(splits)} folds", flush=True)
    return preds


def repeated_scores(folds, y):
    scores, mses = [], []
    for repeat in range(3):
        actual, predicted = [], []
        for fold in range(repeat * 5, (repeat + 1) * 5):
            idx, pred = folds[fold]
            actual.extend(y[idx])
            predicted.extend(pred)
        scores.append(r2_score(actual, predicted))
        mses.append(mean_squared_error(actual, predicted))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), float(np.mean(mses))


def main():
    train_raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(train_raw)
    y = data[TARGET].to_numpy(dtype=float)
    X = make_extended_features(data)
    X, cats = prep_cat(X)
    X.attrs["cat_cols"] = cats
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    splits = list(cv.split(X))

    base_oof = fit_base_oof(X, y, splits)
    base_score = repeated_scores(base_oof, y)
    print(f"Exact 0.91231 pipeline CV: R2={base_score[0]:.5f} +/- {base_score[1]:.5f}", flush=True)

    cat_oof = {n: [] for n in ITERATION_COUNTS}
    for fold, (tr, va) in enumerate(splits, start=1):
        for iterations in ITERATION_COUNTS:
            model = CatBoostRegressor(
                loss_function="RMSE", iterations=iterations, depth=5,
                learning_rate=0.03, l2_leaf_reg=7,
                random_seed=SEED, verbose=False,
                allow_writing_files=False, thread_count=4,
            )
            model.fit(X.iloc[tr], y[tr], cat_features=cats)
            cat_oof[iterations].append((va, model.predict(X.iloc[va])))
        if fold % 5 == 0:
            print(f"Longer CatBoost OOF: {fold}/{len(splits)} folds", flush=True)

    rows = [{
        "candidate": "Exact current pipeline (0.91231 submission)",
        "iterations": 900, "extra_cat_weight": 0.0,
        "mean_repeat_r2": base_score[0], "std_repeat_r2": base_score[1],
        "mean_repeat_mse": base_score[2],
    }]
    for iterations, cat_folds in cat_oof.items():
        direct = repeated_scores(cat_folds, y)
        rows.append({
            "candidate": f"CatBoost only depth5 lr=.03 l2=7 iter={iterations}",
            "iterations": iterations, "extra_cat_weight": 1.0,
            "mean_repeat_r2": direct[0], "std_repeat_r2": direct[1],
            "mean_repeat_mse": direct[2],
        })
        for weight in CAT_WEIGHTS:
            blended = []
            for (idx, current), (cat_idx, cat_pred) in zip(base_oof, cat_folds):
                assert np.array_equal(idx, cat_idx)
                blended.append((idx, (1 - weight) * current + weight * cat_pred))
            score = repeated_scores(blended, y)
            rows.append({
                "candidate": f"Current + CatBoost iter={iterations}; weight={weight:.2f}",
                "iterations": iterations, "extra_cat_weight": weight,
                "mean_repeat_r2": score[0], "std_repeat_r2": score[1],
                "mean_repeat_mse": score[2],
            })

    result = pd.DataFrame(rows).sort_values("mean_repeat_r2", ascending=False)
    output = HERE / "results" / "exact_blend_refinement_cv.csv"
    result.to_csv(output, index=False)
    print("\nTop candidates:")
    print(result.head(12).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
