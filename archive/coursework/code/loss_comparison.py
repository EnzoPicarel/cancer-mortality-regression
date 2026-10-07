"""Controlled MSE-versus-Huber comparison for HW1 Question 10.

The estimator, SGD optimizer, split, preprocessing and hyperparameters are
identical; only SGDRegressor.loss changes. Uses train.csv labels only.
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_models import HERE, ID, RANDOM_SEED, TARGET, correct_median_age, make_preprocessor


def main():
    data, _ = correct_median_age(pd.read_csv(HERE / "data" / "train.csv"))
    X = data.drop(columns=[TARGET, ID, "Geography"], errors="ignore")
    y = data[TARGET].astype(float)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    X_valid, X_local_test, y_valid, y_local_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=RANDOM_SEED
    )

    preprocessor = make_preprocessor(X_train)
    x_train = preprocessor.fit_transform(X_train)
    x_valid = preprocessor.transform(X_valid)
    x_local_test = preprocessor.transform(X_local_test)
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    y_train_scaled = (y_train - target_mean) / target_scale

    # Match all SGDRegressor settings; Huber uses epsilon=0.1 by default.
    common = dict(max_iter=3000, tol=1e-4, random_state=RANDOM_SEED)
    models = {
        "SGD squared_error (MSE)": SGDRegressor(loss="squared_error", **common),
        "SGD Huber (epsilon=0.1)": SGDRegressor(loss="huber", epsilon=0.1, **common),
    }
    results = []
    for name, model in models.items():
        model.fit(x_train, y_train_scaled)
        row = {"model": name, "epochs": int(model.n_iter_)}
        for split_name, features, labels in [
            ("train", x_train, y_train),
            ("validation", x_valid, y_valid),
            ("local_test", x_local_test, y_local_test),
        ]:
            predictions = model.predict(features) * target_scale + target_mean
            row[f"{split_name}_MSE"] = mean_squared_error(labels, predictions)
            row[f"{split_name}_R2"] = r2_score(labels, predictions)
        results.append(row)

    table = pd.DataFrame(results)
    table.to_csv(HERE / "results" / "loss_comparison.csv", index=False)
    print("Same 70/15/15 split (1705/366/366), preprocessing, SGDRegressor settings; loss only differs:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nWrote loss_comparison.csv")


if __name__ == "__main__":
    main()
