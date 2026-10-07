"""Repeated-CV model and blending search using train.csv labels only."""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from train_models import TARGET, ID, correct_median_age
from model_search import make_features, make_pipeline
from feature_search import engineered

HERE = Path(__file__).resolve().parent.parent
SEED = 9351


def tree_pipeline(X, model):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")
    return Pipeline([("prep", prep), ("model", model)])


def main():
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(raw)
    y = data[TARGET].astype(float).to_numpy()
    X = engineered(data, "ratios")
    # County IDs and county names are excluded; the state parsed from Geography repeats.
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    models = {
        "Ridge ratios a=.75": make_pipeline(X, Ridge(alpha=.75)),
        "Ridge ratios a=.75 (state excluded)": make_pipeline(engineered(data, "ratios").drop(columns="state"), Ridge(alpha=.75)),
        "ExtraTrees leaf2": tree_pipeline(X, ExtraTreesRegressor(n_estimators=500, max_features=.85, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
        "ExtraTrees leaf5": tree_pipeline(X, ExtraTreesRegressor(n_estimators=500, max_features=.85, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "RandomForest leaf5": tree_pipeline(X, RandomForestRegressor(n_estimators=500, max_features=.85, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "HistGB leaves7": tree_pipeline(X, HistGradientBoostingRegressor(max_iter=350, learning_rate=.04, max_leaf_nodes=7, l2_regularization=3, min_samples_leaf=18, random_state=SEED)),
        "HistGB leaves15": tree_pipeline(X, HistGradientBoostingRegressor(max_iter=350, learning_rate=.04, max_leaf_nodes=15, l2_regularization=3, min_samples_leaf=18, random_state=SEED)),
        "GradBoost depth2": tree_pipeline(X, GradientBoostingRegressor(n_estimators=500, learning_rate=.025, max_depth=2, min_samples_leaf=8, loss="huber", random_state=SEED)),
    }
    # Save aligned out-of-fold predictions for unbiased fold-level model blends.
    predictions = {k: [] for k in models}
    scores = {k: [] for k in models}
    fold_ids = []
    for fold, (tr, va) in enumerate(cv.split(X), start=1):
        fold_ids.append(va)
        fold_pred = {}
        for name, estimator in models.items():
            est = clone(estimator)
            if name.endswith("state excluded"):
                Xmodel = X.drop(columns="state")
            else:
                Xmodel = X
            est.fit(Xmodel.iloc[tr], y[tr])
            p = est.predict(Xmodel.iloc[va])
            fold_pred[name] = p
            scores[name].append(r2_score(y[va], p))
            predictions[name].append((va, p))
        if fold % 5 == 0:
            print(f"Completed {fold}/15 validation folds", flush=True)

    rows = []
    for name in models:
        s = np.asarray(scores[name])
        rows.append({"model": name, "cv_r2_mean": s.mean(), "cv_r2_std": s.std(ddof=1), "cv_mse_mean": np.mean([mean_squared_error(y[va], p) for va,p in predictions[name]])})
    # Each repeat comprises the same 5 folds. Score blends separately by repeat.
    names = list(models)
    blend_specs = [("ridge + extra trees", names[0], names[2]), ("ridge + HistGB7", names[0], names[5]), ("ridge + HistGB15", names[0], names[6]), ("ridge + GradBoost", names[0], names[7])]
    for blend_name, left, right in blend_specs:
        for weight in [.2, .4, .6, .8]:
            rep_r2, rep_mse = [], []
            for rep in range(3):
                yy, pp = [], []
                for f in range(rep*5, (rep+1)*5):
                    va_l, p_l = predictions[left][f]
                    va_r, p_r = predictions[right][f]
                    assert np.array_equal(va_l, va_r)
                    yy.extend(y[va_l]); pp.extend(weight*p_l+(1-weight)*p_r)
                rep_r2.append(r2_score(yy, pp)); rep_mse.append(mean_squared_error(yy, pp))
            rows.append({"model": f"{blend_name} ridge_weight={weight}", "cv_r2_mean": np.mean(rep_r2), "cv_r2_std": np.std(rep_r2, ddof=1), "cv_mse_mean": np.mean(rep_mse)})
    results = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)
    results.to_csv(HERE / "results" / "ensemble_search_results.csv", index=False)
    print(results.to_string(index=False, float_format=lambda v:f"{v:.4f}"))
    best_name = str(results.iloc[0].model)
    if best_name.startswith("ridge + "):
        blend_name = best_name.split(" ridge_weight=")[0].removeprefix("ridge + ")
        entry = next(item for item in blend_specs if item[0] == blend_name)
        weight = float(best_name.rsplit("=", 1)[1])
        component_names = [entry[1], entry[2]]
        weights = [weight, 1.0 - weight]
    else:
        component_names = [best_name]
        weights = [1.0]
    fitted = []
    for name in component_names:
        estimator = clone(models[name])
        estimator.fit(X, y)
        fitted.append(estimator)
    test, _ = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
    Xt = engineered(test, "ratios")
    pred = sum(weight * estimator.predict(Xt) for weight, estimator in zip(weights, fitted))
    desc = {"model": best_name, "components": component_names, "weights": weights}
    out=pd.DataFrame({ID:test[ID],TARGET:pred})
    out.to_csv(HERE / "results" / "submission_ensemble.csv",index=False)
    joblib.dump({"components": fitted, "component_names": component_names, "weights": weights,
                 "feature_variant": "ratios", "cv_r2_mean": float(results.iloc[0].cv_r2_mean),
                 "cv_r2_std": float(results.iloc[0].cv_r2_std)}, HERE / "models" / "ensemble_model.joblib")
    print("Selected:", desc, "prediction range:", float(pred.min()), float(pred.max()))

if __name__ == "__main__":
    main()
