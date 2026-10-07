"""Plot training-MSE trajectories for all iterative HW1 models.

The DNNs and SGDRegressors are advanced one epoch at a time and evaluated on
the same 70% training partition. Ordinary least squares has no epoch-wise
optimizer; its closed-form training MSE is shown as a point at epoch zero.
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from train_models import (
    HERE, ID, RANDOM_SEED, TARGET, correct_median_age, make_models, make_preprocessor,
)

MAX_EPOCHS = 200

# Each warm-start fit intentionally performs one epoch; sklearn's warning that
# a single epoch is not converged is expected in this curve-recording loop.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.neural_network")


def main():
    data, _ = correct_median_age(pd.read_csv(HERE / "data" / "train.csv"))
    X = data.drop(columns=[TARGET, ID, "Geography"], errors="ignore")
    y = data[TARGET].astype(float)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    preprocessor = make_preprocessor(X_train)
    x_train = preprocessor.fit_transform(X_train)
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std()) or 1.0
    y_scaled = (y_train.to_numpy() - target_mean) / target_scale

    histories = {}
    dnn_names = [
        "DNN-8-output", "DNN-16-8-output", "DNN-16-8-4-output",
        "DNN-30-16-8-4-output", "DNN-64-32-output (extra)",
    ]
    dnn_templates = make_models()
    for idx, name in enumerate(dnn_names):
        model = dnn_templates[name]
        model.set_params(
            max_iter=1, warm_start=True, early_stopping=False, tol=0.0,
            n_iter_no_change=MAX_EPOCHS + 1,
            random_state=np.random.RandomState(RANDOM_SEED + idx),
        )
        curve = []
        for _ in range(MAX_EPOCHS):
            model.fit(x_train, y_scaled)
            prediction = model.predict(x_train) * target_scale + target_mean
            curve.append(mean_squared_error(y_train, prediction))
        histories[name] = np.asarray(curve)

    # Record an epoch-wise common MSE metric for the two controlled loss models.
    for name, loss, seed in [
        ("SGD squared-error (Q10)", "squared_error", 42),
        ("SGD Huber (Q10)", "huber", 42),
    ]:
        model = SGDRegressor(
            loss=loss, max_iter=3000, tol=1e-4, random_state=np.random.RandomState(seed),
            **({"epsilon": 0.1} if loss == "huber" else {}),
        )
        curve = []
        for _ in range(MAX_EPOCHS):
            model.partial_fit(x_train, y_scaled)
            prediction = model.predict(x_train) * target_scale + target_mean
            curve.append(mean_squared_error(y_train, prediction))
        histories[name] = np.asarray(curve)

    # LinearRegression solves least squares directly instead of iterating.
    linear = LinearRegression().fit(x_train, y_train)
    linear_mse = mean_squared_error(y_train, linear.predict(x_train))

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.get_cmap("tab10")
    epochs = np.arange(1, MAX_EPOCHS + 1)
    for idx, (name, values) in enumerate(histories.items()):
        ax.plot(epochs, values, label=name, linewidth=1.7, color=colors(idx))
    ax.scatter([0], [linear_mse], marker="D", s=75, color="black", zorder=5,
               label="Linear regression (closed-form fit; no epochs)")
    ax.set_title("Training-loss trajectories on the common training split")
    ax.set_xlabel("Epoch (one pass through the training split)")
    ax.set_ylabel("Training MSE (original death-rate units squared)")
    ax.set_xlim(-4, MAX_EPOCHS + 2)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "training_loss_all_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote training_loss_all_models.png ({len(histories)} curves + linear point).")
    print(f"LinearRegression closed-form reference training MSE: {linear_mse:.4f}")


if __name__ == "__main__":
    main()
