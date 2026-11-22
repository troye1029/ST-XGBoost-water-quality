"""Recursive feature elimination, Optuna tuning and XGBoost training."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import optuna
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

LOGGER = logging.getLogger("st-xgboost")


PARAM_SPACE = {
    "n_estimators": ("int", 200, 500),
    "max_depth": ("int", 4, 8),
    "learning_rate": ("float", 0.03, 0.15),
    "subsample": ("float", 0.6, 0.9),
    "colsample_bytree": ("float", 0.6, 0.9),
    "reg_alpha": ("float", 1.0, 10.0),
    "reg_lambda": ("float", 1.0, 10.0),
    "min_child_weight": ("int", 5, 20),
}


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> Dict[str, float]:
    """Compute the error metrics reported in the paper.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(
            np.mean(
                np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
            )
            * 100.0
        )
    else:
        mape = float("nan")

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = denominator != 0
    if valid.any():
        smape = float(
            np.mean(
                np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]
            )
            * 100.0
        )
    else:
        smape = float("nan")

    try:
        msle = float(
            mean_squared_log_error(
                np.maximum(y_true, 0.0), np.maximum(y_pred, 0.0)
            )
        )
    except ValueError:
        msle = float("nan")

    medae = float(median_absolute_error(y_true, y_pred))

    delta = 1.0
    error = y_true - y_pred
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    huber_loss = float(
        np.where(np.abs(error) <= delta, squared_loss, linear_loss).mean()
    )

    return {
        "Model_Set": label,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": smape,
        "R2": r2,
        "MSLE": msle,
        "MedAE": medae,
        "Huber_Loss": huber_loss,
        "N": int(len(y_true)),
    }


def select_features(
    X: np.ndarray, y: np.ndarray, n_features: int, seed: int, n_jobs: int
) -> np.ndarray:
    """Run recursive feature elimination with an XGBoost base learner."""
    n_features = int(min(n_features, X.shape[1]))
    LOGGER.info(
        "RFE: selecting %d of %d predictors ...", n_features, X.shape[1]
    )

    estimator = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    selector.fit(X, y)
    return selector.support_


def _suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, float]:
    """Draw one hyperparameter set from the search space of the paper."""
    params: Dict[str, float] = {}
    for name, (kind, low, high) in PARAM_SPACE.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    seed: int,
    cv_splits: int,
    n_jobs: int,
) -> optuna.Study:
    """Tune the XGBoost hyperparameters with Optuna/TPE. """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hyperparameters(trial)
        model_params = dict(
            params,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )

        if cv_splits >= 2:
            splitter = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            for fit_index, validation_index in splitter.split(X):
                model = xgb.XGBRegressor(**model_params)
                model.fit(X[fit_index], y[fit_index], verbose=False)
                prediction = model.predict(X[validation_index])
                scores.append(r2_score(y[validation_index], prediction))
            return float(np.mean(scores))

        cut = int(len(X) * 0.8)
        model = xgb.XGBRegressor(**model_params)
        model.fit(X[:cut], y[:cut], verbose=False)
        prediction = model.predict(X[cut:])
        return float(r2_score(y[cut:], prediction))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    LOGGER.info("Optuna best score (R2): %.4f", study.best_value)
    LOGGER.info("Optuna best hyperparameters: %s", study.best_params)
    return study


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, float],
    seed: int,
    n_jobs: int,
) -> xgb.XGBRegressor:
    """Fit the tuned model on the whole training period."""
    model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    return model
