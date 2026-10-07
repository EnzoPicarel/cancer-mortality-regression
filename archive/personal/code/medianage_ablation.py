"""Compare raw, month-to-year corrected, and dropped MedianAge variants.

Run from this folder with: python medianage_ablation.py
Uses the exact same seed-42 70/15/15 row split for every variant.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_models import ID, RANDOM_SEED, TARGET, correct_median_age, make_models, make_preprocessor


HERE = Path(__file__).resolve().parent.parent


def features(frame: pd.DataFrame, *, drop_age: bool = False) -> pd.DataFrame:
    excluded = [TARGET, ID, "Geography"]
    if drop_age:
        excluded.append("MedianAge")
    return frame.drop(columns=excluded, errors="ignore")


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    corrected, corrected_count = correct_median_age(raw)
    variants = {
        "raw": (raw, False),
        "divide values >100 by 12": (corrected, False),
        "drop MedianAge": (corrected, True),
    }
    y = raw[TARGET].astype(float)
    indices = pd.Series(range(len(raw)))
    train_i, remainder_i = train_test_split(
        indices, test_size=0.30, random_state=RANDOM_SEED
    )
    valid_i, test_i = train_test_split(
        remainder_i, test_size=0.50, random_state=RANDOM_SEED
    )
    y_train, y_valid, y_test = y.iloc[train_i], y.iloc[valid_i], y.iloc[test_i]
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std()) or 1.0
    y_train_scaled = (y_train - target_mean) / target_scale

    all_models = make_models()
    chosen = {
        name: all_models[name]
        for name in ["Linear regression", "DNN-16-8-output", "SGD regression (Huber loss)"]
    }
    rows = []

    for variant_name, (frame, drop_age) in variants.items():
        X = features(frame, drop_age=drop_age)
        X_train, X_valid, X_test = X.iloc[train_i], X.iloc[valid_i], X.iloc[test_i]
        for model_name, model in chosen.items():
            prep = make_preprocessor(X_train)
            x_train = prep.fit_transform(X_train)
            x_valid = prep.transform(X_valid)
            x_test = prep.transform(X_test)
            model.fit(x_train, y_train_scaled)
            predictions = {
                "train": model.predict(x_train) * target_scale + target_mean,
                "validation": model.predict(x_valid) * target_scale + target_mean,
                "local_test": model.predict(x_test) * target_scale + target_mean,
            }
            actuals = {"train": y_train, "validation": y_valid, "local_test": y_test}
            rows.append(
                {
                    "medianage_variant": variant_name,
                    "model": model_name,
                    "train_MSE": mean_squared_error(actuals["train"], predictions["train"]),
                    "train_R2": r2_score(actuals["train"], predictions["train"]),
                    "validation_MSE": mean_squared_error(actuals["validation"], predictions["validation"]),
                    "validation_R2": r2_score(actuals["validation"], predictions["validation"]),
                    "local_test_MSE": mean_squared_error(actuals["local_test"], predictions["local_test"]),
                    "local_test_R2": r2_score(actuals["local_test"], predictions["local_test"]),
                }
            )
            print(f"Finished {variant_name}: {model_name}")

    results = pd.DataFrame(rows)
    results.to_csv(HERE / "results" / "medianage_ablation.csv", index=False)
    print(f"Corrected {corrected_count} training rows.")
    print("\nAblation results:")
    print(
        results.sort_values(["model", "medianage_variant"])
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nWrote medianage_ablation.csv")


if __name__ == "__main__":
    main()
