"""Diagnostic figures for model evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

LOGGER = logging.getLogger("st-xgboost")


def configure_matplotlib() -> None:
    """Apply publication-oriented figure defaults (English labels only)."""
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Write a figure at 300 dpi and release its memory."""
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(
    target: str,
    model: xgb.XGBRegressor,
    feature_names: Sequence[str],
    y_train: np.ndarray,
    yp_train: np.ndarray,
    y_test: np.ndarray,
    yp_test: np.ndarray,
    study: optuna.Study,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_year: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    """Write the twelve diagnostic figures of the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_test - yp_test
    r2 = float(r2_score(y_test, yp_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, yp_test)))
    test_label = f"test year {test_year}"
    LOGGER.info("  writing 12 diagnostic figures to %s", output_dir)

    # 1. Observed versus predicted
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    axis.scatter(
        y_test, yp_test, s=6, alpha=0.35, color="#1f77b4", linewidths=0
    )
    lower = float(min(y_test.min(), yp_test.min()))
    upper = float(max(y_test.max(), yp_test.max()))
    axis.plot(
        [lower, upper], [lower, upper], color="#d62728", linestyle="--", linewidth=1.2
    )
    axis.set_xlabel("Observed value")
    axis.set_ylabel("Predicted value")
    axis.set_title(f"{target}: observed versus predicted ({test_label})")
    axis.text(
        0.04,
        0.96,
        f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nN = {len(y_test):,}",
        transform=axis.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    _save_figure(figure, output_dir / "01_True_vs_Pred.png")

    # 2. Residual distribution
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    sns.histplot(x=residuals, kde=True, color="#6a3d9a", bins=50, ax=axis)
    axis.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Residual (observed - predicted)")
    axis.set_ylabel("Count")
    axis.set_title(f"{target}: residual distribution ({test_label})")
    _save_figure(figure, output_dir / "02_Residual_Hist.png")

    # 3. Residuals versus predicted
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.scatter(
        yp_test, residuals, s=6, alpha=0.3, color="#2ca02c", linewidths=0
    )
    axis.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Predicted value")
    axis.set_ylabel("Residual")
    axis.set_title(f"{target}: residuals versus predicted ({test_label})")
    _save_figure(figure, output_dir / "03_Residual_Plot.png")

    # 4. Feature importance of the RFE-selected subset
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    order = np.argsort(importances)[::-1]
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.bar(
        np.arange(len(importances)),
        importances[order],
        color="#008080",
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(np.arange(len(importances)))
    axis.set_xticklabels(
        [feature_names[i] for i in order], rotation=45, ha="right"
    )
    axis.set_ylabel("Gain-based importance")
    axis.set_title(f"{target}: feature importance after RFE")
    figure.tight_layout()
    _save_figure(figure, output_dir / "04_Feature_Importance.png")

    # 5. Learning curve (training versus held-out test year)
    evaluation = model.evals_result()
    if evaluation:
        keys = list(evaluation.keys())
        metric = "rmse" if "rmse" in evaluation[keys[0]] else list(evaluation[keys[0]])[0]
        iterations = np.arange(len(evaluation[keys[0]][metric]))
        figure, axis = plt.subplots(figsize=(8.0, 5.0))
        axis.plot(
            iterations, evaluation[keys[0]][metric], label="Training"
        )
        if len(keys) > 1:
            axis.plot(
                iterations,
                evaluation[keys[1]][metric],
                label=f"Held-out {test_label} (diagnostic only)",
            )
        axis.set_xlabel("Boosting iteration")
        axis.set_ylabel(metric.upper())
        axis.set_title(f"{target}: learning curve")
        axis.legend()
        _save_figure(figure, output_dir / "05_Learning_Curve.png")

    # 6. Optuna optimization history
    trial_values = [
        float(trial.value)
        for trial in study.trials
        if trial.value is not None and np.isfinite(trial.value)
    ]
    if trial_values:
        best_so_far = np.maximum.accumulate(trial_values)
        figure, axis = plt.subplots(figsize=(8.0, 5.0))
        axis.plot(
            np.arange(1, len(trial_values) + 1),
            trial_values,
            "o",
            alpha=0.45,
            label="Trial score",
            color="#1f77b4",
        )
        axis.plot(
            np.arange(1, len(trial_values) + 1),
            best_so_far,
            "-",
            linewidth=1.6,
            label="Best so far",
            color="#d62728",
        )
        axis.set_xlabel("Trial")
        axis.set_ylabel("Hold-out R2")
        axis.set_title(f"{target}: Optuna optimization history")
        axis.legend()
        if len(trial_values) <= 20:
            axis.set_xticks(np.arange(1, len(trial_values) + 1))
        _save_figure(figure, output_dir / "06_Optimization_History.png")

    # 7. Fitting detail for 100 randomly drawn test cells
    n_detail = min(100, len(y_test))
    sample = rng.choice(len(y_test), size=n_detail, replace=False)
    order = np.argsort(y_test[sample])
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.plot(
        np.arange(n_detail),
        y_test[sample][order],
        "o-",
        color="#1f77b4",
        markersize=3,
        label="Observed",
        alpha=0.8,
    )
    axis.plot(
        np.arange(n_detail),
        yp_test[sample][order],
        "x--",
        color="#d62728",
        markersize=3,
        label="Predicted",
        alpha=0.8,
    )
    axis.set_xlabel("Sampled test cells (sorted by observed value)")
    axis.set_ylabel(target)
    axis.set_title(f"{target}: fitting detail ({n_detail} cells)")
    axis.legend()
    _save_figure(figure, output_dir / "07_Fitting_Curve.png")

    # 8. Q-Q plot of the residuals
    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    stats.probplot(residuals, dist="norm", plot=axis)
    axis.set_title(f"{target}: Q-Q plot of residuals")
    _save_figure(figure, output_dir / "08_QQ_Plot.png")

    # 9. Absolute error by year (training in-sample, test year held out)
    error_frame = pd.DataFrame(
        {
            "Year": np.concatenate(
                [train_frame["year"].to_numpy(), test_frame["year"].to_numpy()]
            ),
            "Absolute error": np.concatenate(
                [np.abs(y_train - yp_train), np.abs(residuals)]
            ),
            "Subset": np.concatenate(
                [
                    np.full(len(y_train), "Training (in-sample)"),
                    np.full(len(y_test), "Test year (held-out)"),
                ]
            ),
        }
    )
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    sns.boxplot(
        x="Year",
        y="Absolute error",
        hue="Subset",
        data=error_frame,
        palette={
            "Training (in-sample)": "#9ecae1",
            "Test year (held-out)": "#d62728",
        },
        fliersize=1,
        linewidth=0.8,
        ax=axis,
    )
    axis.set_title(f"{target}: absolute error by year")
    axis.legend(title="")
    _save_figure(figure, output_dir / "09_Error_by_Year.png")

    # 10. Spatial distribution of the test residuals
    figure, axis = plt.subplots(figsize=(9.0, 6.5))
    limit = float(np.percentile(np.abs(residuals), 95))
    scatter = axis.scatter(
        test_frame["longitude"],
        test_frame["latitude"],
        c=residuals,
        cmap="seismic",
        s=2,
        alpha=0.8,
        vmin=-limit,
        vmax=limit,
        edgecolors="none",
    )
    figure.colorbar(scatter, ax=axis, label="Residual (observed - predicted)")
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (degrees north)")
    axis.set_title(f"{target}: spatial distribution of residuals ({test_label})")
    axis.set_aspect("equal")
    _save_figure(figure, output_dir / "10_Spatial_Error_Map.png")

    # 11. Distribution comparison
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    sns.kdeplot(
        x=y_train,
        fill=True,
        color="#7f7f7f",
        alpha=0.25,
        label="Training observed",
        ax=axis,
    )
    sns.kdeplot(
        x=y_test,
        fill=True,
        color="#1f77b4",
        alpha=0.3,
        label=f"Test observed ({test_year})",
        ax=axis,
    )
    sns.kdeplot(
        x=yp_test,
        fill=True,
        color="#d62728",
        alpha=0.3,
        label="Test predicted",
        ax=axis,
    )
    axis.set_xlabel(target)
    axis.set_ylabel("Density")
    axis.set_title(f"{target}: distribution comparison")
    axis.legend()
    _save_figure(figure, output_dir / "11_Distribution_Compare.png")

    # 12. Correlation matrix of the selected predictors
    correlation = test_frame[list(feature_names)].corr()
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    sns.heatmap(
        correlation,
        annot=len(feature_names) <= 20,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        ax=axis,
    )
    axis.set_title(f"{target}: predictor correlation ({test_label})")
    figure.tight_layout()
    _save_figure(figure, output_dir / "12_Feature_Correlation.png")


def plot_r2_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write the baseline versus framework R2 comparison bar chart."""
    labels = summary["Target"].tolist()
    positions = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    baseline_bars = axis.bar(
        positions - width / 2,
        summary["Baseline_R2"],
        width,
        label="XGBoost baseline",
        color="#aec7e8",
        edgecolor="black",
        linewidth=0.5,
    )
    framework_bars = axis.bar(
        positions + width / 2,
        summary["Framework_R2"],
        width,
        label="RFE-Optuna-XGBoost",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )

    for bars in (baseline_bars, framework_bars):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("$R^2$ on the held-out test year")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Baseline versus adaptive framework")
    axis.legend(loc="upper right")
    figure.tight_layout()
    _save_figure(figure, output_dir / "00_Baseline_vs_Framework_R2_Comparison.png")
