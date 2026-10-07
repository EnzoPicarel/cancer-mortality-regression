"""Train and evaluate the Homework 1 linear and DNN models.

Run from this folder with:
    python train_models.py

Dependencies: pandas and scikit-learn.
Only train.csv is used. Kaggle submission generation is handled separately by
final_tuning.py so this assignment run cannot overwrite submission.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
TARGET = "TARGET_deathRate"
ID = "id"


def correct_median_age(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert apparent month values in MedianAge to years, preserving other rows."""
    result = frame.copy()
    if "MedianAge" not in result.columns:
        return result, 0
    month_rows = result["MedianAge"] > 100
    count = int(month_rows.sum())
    result.loc[month_rows, "MedianAge"] = result.loc[month_rows, "MedianAge"] / 12.0
    return result, count


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing from feature types; fit it only on each training split."""
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in frame.columns if c not in categorical and c != ID]

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models() -> dict[str, object]:
    """Required linear/DNN models, one extra architecture, and Huber loss run."""
    # The assignment requests SGD with default MLPRegressor settings. Set the
    # solver and a fixed seed only; leave the remaining optimizer parameters
    # (learning rate, batch size, max_iter, early stopping, etc.) at defaults.
    common = dict(solver="sgd", random_state=RANDOM_SEED)
    models = {"Linear regression": LinearRegression()}
    for name, layers in [
        ("DNN-8-output", (8,)),
        ("DNN-16-8-output", (16, 8)),
        ("DNN-16-8-4-output", (16, 8, 4)),
        ("DNN-30-16-8-4-output", (30, 16, 8, 4)),
        ("DNN-64-32-output (extra)", (64, 32)),
    ]:
        models[name] = MLPRegressor(hidden_layer_sizes=layers, **common)

    # Keep an Huber SGD model among the homework comparisons. The controlled
    # same-estimator MSE-vs-Huber comparison is in loss_comparison.py.
    models["SGD regression (Huber loss)"] = SGDRegressor(
        loss="huber", max_iter=3000, tol=1e-4, random_state=RANDOM_SEED
    )
    return models


def main() -> None:
    train_path = HERE / "data" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Place train.csv beside this script in hw1/.")

    raw_data = pd.read_csv(train_path)
    data, corrected_train_count = correct_median_age(raw_data)
    print(f"Converted {corrected_train_count} MedianAge values from months to years in train.csv.")
    # Each Geography occurs exactly once in this file. One-hot encoding it
    # gives linear regression a separate coefficient per row and lets it
    # memorize the training targets, so exclude this identifier-like text.
    X = data.drop(columns=[TARGET, ID, "Geography"], errors="ignore")
    y = data[TARGET].astype(float)

    # 70/15/15 split of the labeled data: the final split is held out until
    # model comparison is complete. Kaggle test labels are never available.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    X_valid, X_local_test, y_valid, y_local_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=RANDOM_SEED
    )
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std()) or 1.0
    y_train_scaled = (y_train - target_mean) / target_scale

    print(f"Split sizes: train={len(X_train)}, validation={len(X_valid)}, "
          f"local test={len(X_local_test)}")
    models = make_models()
    rows = []

    for name, model in models.items():
        prep = make_preprocessor(X_train)
        xtr = prep.fit_transform(X_train)
        xval = prep.transform(X_valid)
        xtest = prep.transform(X_local_test)

        model.fit(xtr, y_train_scaled)
        pred_train = model.predict(xtr) * target_scale + target_mean
        pred_valid = model.predict(xval) * target_scale + target_mean
        pred_test = model.predict(xtest) * target_scale + target_mean

        rows.append(
            {
                "model": name,
                "train_MSE": mean_squared_error(y_train, pred_train),
                "train_R2": r2_score(y_train, pred_train),
                "validation_MSE": mean_squared_error(y_valid, pred_valid),
                "validation_R2": r2_score(y_valid, pred_valid),
                "local_test_MSE": mean_squared_error(y_local_test, pred_test),
                "local_test_R2": r2_score(y_local_test, pred_test),
            }
        )
        print(f"Finished: {name}")

    results = pd.DataFrame(rows).sort_values("validation_R2", ascending=False)
    results.to_csv(HERE / "results" / "model_results.csv", index=False)
    print("\nResults (selection uses validation; local test is for final reporting):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFor the report, use local_test_MSE and local_test_R2 as held-out estimates.")


if __name__ == "__main__":
    main()
