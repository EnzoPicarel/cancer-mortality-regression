"""Feature and spline preprocessing search for the personal Kaggle copy."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent.parent

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ensemble_search import tree_pipeline
from feature_search import STATE_TO_DIVISION, engineered
from model_search import make_pipeline
from train_models import TARGET, correct_median_age

SEED = 9351


def make_extended_features(frame):
    X = engineered(frame, "ratios")
    state = frame["Geography"].astype(str).str.rsplit(",", n=1).str[-1].str.strip()
    X["division"] = state.map(STATE_TO_DIVISION).fillna("Unknown")

    male_age = frame["MedianAgeMale"]
    female_age = frame["MedianAgeFemale"]
    X["age_gender_gap"] = female_age - male_age
    X["overall_vs_gender_age"] = frame["MedianAge"] - (male_age + female_age) / 2

    private = frame["PctPrivateCoverage"]
    public = frame["PctPublicCoverage"]
    X["private_minus_public_coverage"] = private - public
    X["private_to_public_coverage"] = private / (public + 1.0)
    X["private_coverage_share"] = private / (private + public + 1.0)
    X["coverage_alone_share"] = frame["PctPrivateCoverageAlone"] / (private + 1.0)

    X["income_after_poverty_adjustment"] = frame["medIncome"] * (1 - frame["povertyPercent"] / 100)
    X["income_per_poverty_point"] = frame["medIncome"] / (frame["povertyPercent"] + 1.0)
    X["education_degree_minus_no_hs"] = frame["PctBachDeg25_Over"] - frame["PctNoHS18_24"]
    X["degree_share_over_hs"] = frame["PctBachDeg25_Over"] / (frame["PctHS25_Over"] + 1.0)
    X["married_household_gap"] = frame["PctMarriedHouseholds"] - frame["PercentMarried"]
    X["employment_to_unemployment"] = frame["PctEmployed16_Over"] / (frame["PctUnemployed16_Over"] + 1.0)

    # The income-bin midpoint adds a continuous version of the existing interval label.
    bounds = frame["binnedInc"].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)[^0-9]+([0-9]+(?:\.[0-9]+)?)")
    X["income_bin_midpoint"] = (pd.to_numeric(bounds[0], errors="coerce") + pd.to_numeric(bounds[1], errors="coerce")) / 2

    for column in ["medIncome", "incidenceRate", "deaths_per_case",
                   "cases_per_100k_population", "deaths_per_100k_population"]:
        X[f"log1p_{column}"] = np.log1p(X[column].clip(lower=0))
    return X


def make_spline_ridge(X, n_knots, alpha):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=n_knots, degree=3, knots="quantile", include_bias=False)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric_splines", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([("preprocess", prep), ("model", Ridge(alpha=alpha))])


def catboost_model(depth=5):
    return CatBoostRegressor(
        loss_function="RMSE", iterations=900, depth=depth, learning_rate=0.035,
        l2_leaf_reg=5, random_seed=SEED, verbose=False,
        allow_writing_files=False, thread_count=4,
    )


def lightgbm_model():
    return LGBMRegressor(
        objective="regression", n_estimators=700, num_leaves=15,
        learning_rate=0.03, min_child_samples=20, reg_lambda=5,
        random_state=SEED, verbosity=-1, n_jobs=4,
        deterministic=True, force_col_wise=True,
    )


def fit_predict(spec, X_train, y_train, X_valid):
    kind, model = spec
    if kind == "catboost":
        cats = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        tr, va = X_train.copy(), X_valid.copy()
        for col in cats:
            tr[col] = tr[col].fillna("Unknown").astype(str)
            va[col] = va[col].fillna("Unknown").astype(str)
        model.fit(tr, y_train, cat_features=cats)
        return model.predict(va)
    if kind == "lightgbm":
        cats = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        tr, va = X_train.copy(), X_valid.copy()
        for col in cats:
            categories = pd.Index(pd.concat([tr[col], va[col]], ignore_index=True).fillna("Unknown").astype(str).unique())
            tr[col] = pd.Categorical(tr[col].fillna("Unknown").astype(str), categories=categories)
            va[col] = pd.Categorical(va[col].fillna("Unknown").astype(str), categories=categories)
        model.fit(tr, y_train, categorical_feature=cats)
        return model.predict(va)
    if kind == "current_blend":
        ridge = make_pipeline(X_train, Ridge(alpha=0.75)).fit(X_train, y_train)
        hist = tree_pipeline(
            X_train,
            HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            ),
        ).fit(X_train, y_train)
        return 0.35 * ridge.predict(X_valid) + 0.65 * hist.predict(X_valid)
    model.fit(X_train, y_train)
    return model.predict(X_valid)


def repeat_scores(name, folds, y):
    result = []
    for repeat in range(3):
        actual, predicted = [], []
        for fold in range(repeat * 5, (repeat + 1) * 5):
            idx, pred = folds[fold]
            actual.extend(y[idx])
            predicted.extend(pred)
        result.append((r2_score(actual, predicted), mean_squared_error(actual, predicted)))
    return result


def main():
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(raw)
    y = data[TARGET].to_numpy(dtype=float)
    X_base = engineered(data, "ratios")
    X_ext = make_extended_features(data)

    candidates = {
        "Current 35/65 blend, original ratios": ("current_blend", None, X_base),
        "Ridge alpha=.75, extended features": ("pipeline", make_pipeline(X_ext, Ridge(alpha=0.75)), X_ext),
        "HistGB, extended features": ("pipeline", tree_pipeline(
            X_ext, HistGradientBoostingRegressor(
                max_leaf_nodes=15, learning_rate=0.04, max_iter=350,
                min_samples_leaf=12, l2_regularization=3, random_state=SEED,
            )), X_ext),
        "Spline Ridge 4 knots alpha=10, ratios": ("pipeline", make_spline_ridge(X_base, 4, 10), X_base),
        "Spline Ridge 4 knots alpha=10, extended": ("pipeline", make_spline_ridge(X_ext, 4, 10), X_ext),
        "Spline Ridge 5 knots alpha=30, extended": ("pipeline", make_spline_ridge(X_ext, 5, 30), X_ext),
        "CatBoost depth=5, extended features": ("catboost", catboost_model(5), X_ext),
        "LightGBM leaves=15, extended features": ("lightgbm", lightgbm_model(), X_ext),
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    splits = list(cv.split(X_base))
    predictions = {name: [] for name in candidates}
    fold_r2 = {name: [] for name in candidates}
    for fold, (tr, va) in enumerate(splits, start=1):
        for name, (kind, model, X) in candidates.items():
            pred = fit_predict((kind, model), X.iloc[tr], y[tr], X.iloc[va])
            predictions[name].append((va, pred))
            fold_r2[name].append(r2_score(y[va], pred))
        if fold % 5 == 0:
            print(f"Completed {fold}/{len(splits)} folds", flush=True)

    rows = []
    score_map = {}
    for name in candidates:
        reps = repeat_scores(name, predictions[name], y)
        score_map[name] = reps
        rows.append({
            "candidate": name,
            "mean_fold_r2": float(np.mean(fold_r2[name])),
            "std_fold_r2": float(np.std(fold_r2[name], ddof=1)),
            "mean_repeat_r2": float(np.mean([score[0] for score in reps])),
            "std_repeat_r2": float(np.std([score[0] for score in reps], ddof=1)),
            "mean_repeat_mse": float(np.mean([score[1] for score in reps])),
        })

    baseline = "Current 35/65 blend, original ratios"
    for name in list(candidates):
        if name == baseline:
            continue
        for weight in [0.2, 0.4, 0.6, 0.8]:
            blended = []
            for fold, (idx, other_pred) in enumerate(predictions[name]):
                base_idx, base_pred = predictions[baseline][fold]
                assert np.array_equal(idx, base_idx)
                blended.append((idx, (1 - weight) * base_pred + weight * other_pred))
            reps = repeat_scores(f"{name} blend", blended, y)
            rows.append({
                "candidate": f"Blend current + {name}; other_weight={weight:.1f}",
                "mean_fold_r2": np.nan,
                "std_fold_r2": np.nan,
                "mean_repeat_r2": float(np.mean([score[0] for score in reps])),
                "std_repeat_r2": float(np.std([score[0] for score in reps], ddof=1)),
                "mean_repeat_mse": float(np.mean([score[1] for score in reps])),
            })

    results = pd.DataFrame(rows).sort_values("mean_repeat_r2", ascending=False)
    out = HERE / "results" / "preprocessing_cv_comparison.csv"
    results.to_csv(out, index=False)
    print("\nRepeated 5-fold CV results (3 repeats):")
    print(results.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
