"""Search a few domain-driven feature variants using repeated CV on train.csv."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from model_search import make_features, make_pipeline
from advanced_search import INTERACTION_FIELDS
from train_models import ID, RANDOM_SEED, TARGET, correct_median_age


HERE = Path(__file__).resolve().parent.parent
STATE_TO_DIVISION = {
    **dict.fromkeys(["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"], "New England"),
    **dict.fromkeys(["New Jersey", "New York", "Pennsylvania"], "Middle Atlantic"),
    **dict.fromkeys(["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"], "East North Central"),
    **dict.fromkeys(["Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"], "West North Central"),
    **dict.fromkeys(["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia"], "South Atlantic"),
    **dict.fromkeys(["Alabama", "Kentucky", "Mississippi", "Tennessee"], "East South Central"),
    **dict.fromkeys(["Arkansas", "Louisiana", "Oklahoma", "Texas"], "West South Central"),
    **dict.fromkeys(["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming"], "Mountain"),
    **dict.fromkeys(["Alaska", "California", "Hawaii", "Oregon", "Washington"], "Pacific"),
}


def engineered(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    use_state = variant not in {"division only"}
    X = make_features(frame, add_state=use_state)
    if variant in {"state + division", "division only"}:
        state = frame["Geography"].astype(str).str.rsplit(",", n=1).str[-1].str.strip()
        X["division"] = state.map(STATE_TO_DIVISION).fillna("Unknown")
    if variant in {"ratios", "ratios + interactions"}:
        deaths = frame["avgDeathsPerYear"].clip(lower=0)
        cases = frame["avgAnnCount"].clip(lower=0)
        population = frame["popEst2015"].clip(lower=1)
        X["deaths_per_case"] = deaths / cases.clip(lower=1)
        X["cases_per_100k_population"] = cases / population * 100_000
        X["deaths_per_100k_population"] = deaths / population * 100_000
    if variant == "id numeric":
        X["id_numeric"] = frame[ID]
    if variant == "drop youth education":
        X = X.drop(columns=["PctSomeCol18_24", "PctHS18_24", "PctNoHS18_24", "PctBachDeg18_24"], errors="ignore")
    if variant == "ratios + interactions":
        # Add the ratios to the pre-specified interaction block.
        pass
    return X


def poly_pipeline(X: pd.DataFrame, alpha: float) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    core = [c for c in INTERACTION_FIELDS if c in numeric]
    core += [c for c in ["deaths_per_case", "cases_per_100k_population", "deaths_per_100k_population"] if c in numeric]
    rest = [c for c in numeric if c not in core]
    transforms = []
    if core:
        transforms.append(("interactions", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ]), core))
    if rest:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), rest))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    pre = ColumnTransformer(transforms, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", Ridge(alpha=alpha))])


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    data, _ = correct_median_age(raw)
    y = data[TARGET].astype(float)
    variants = ["baseline state + logs", "ratios", "state + division", "division only", "id numeric", "drop youth education", "ratios + interactions"]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=31415)
    rows = []

    for variant in variants:
        X = engineered(data, variant)
        configs = []
        if variant == "ratios + interactions":
            configs = [(f"Ridge interactions alpha={a}", poly_pipeline(X, a)) for a in [0.1, 0.3, 0.75, 1.5]]
        else:
            configs = [(f"Ridge alpha={a}", make_pipeline(X, Ridge(alpha=a))) for a in [0.1, 0.3, 0.75, 1.5, 3.0]]
        for model_name, estimator in configs:
            scores = cross_validate(
                estimator, X, y, cv=cv,
                scoring={"r2": "r2", "mse": "neg_mean_squared_error"}, n_jobs=1,
            )
            row = {
                "feature_variant": variant,
                "model": model_name,
                "cv_R2_mean": scores["test_r2"].mean(),
                "cv_R2_std": scores["test_r2"].std(ddof=1),
                "cv_MSE_mean": -scores["test_mse"].mean(),
                "cv_MSE_std": scores["test_mse"].std(ddof=1),
            }
            rows.append(row)
            print(f"{variant:27s} {model_name:32s} R2={row['cv_R2_mean']:.4f} +/- {row['cv_R2_std']:.4f}")

    results = pd.DataFrame(rows).sort_values("cv_R2_mean", ascending=False)
    results.to_csv(HERE / "results" / "feature_search_results.csv", index=False)
    print("\nTop configurations:")
    print(results.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best = results.iloc[0]
    variant = str(best.feature_variant)
    alpha = float(best.model.rsplit("=", 1)[1])
    X = engineered(data, variant)
    if variant == "ratios + interactions":
        final = poly_pipeline(X, alpha)
    else:
        final = make_pipeline(X, Ridge(alpha=alpha))
    final.fit(X, y)
    test, _ = correct_median_age(pd.read_csv(HERE / "data" / "test.csv"))
    X_test = engineered(test, variant)
    sub = pd.DataFrame({ID: test[ID], TARGET: final.predict(X_test)})
    sub.to_csv(HERE / "results" / "submission_feature_search.csv", index=False)
    joblib.dump({"pipeline": final, "feature_variant": variant, "alpha": alpha,
                 "cv_R2_mean": float(best.cv_R2_mean), "cv_R2_std": float(best.cv_R2_std)},
                HERE / "models" / "feature_search_model.joblib")
    print(f"\nBest: {variant}, {best.model}, CV R2={best.cv_R2_mean:.4f} +/- {best.cv_R2_std:.4f}")
    print(f"Wrote submission_feature_search.csv ({len(sub)} rows; range {sub[TARGET].min():.2f}–{sub[TARGET].max():.2f}).")


if __name__ == "__main__":
    main()
