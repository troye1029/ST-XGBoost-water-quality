from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from features import (
    K_BACKGROUND,
    TARGETS,
    add_spatial_background,
    build_feature_table,
    feature_names_for,
    load_dataset,
    split_train_test,
)
from modeling import (
    regression_metrics,
    select_features,
    train_final_model,
    tune_hyperparameters,
)
from plotting import (
    configure_matplotlib,
    plot_diagnostics,
    plot_r2_comparison,
)

LOGGER = logging.getLogger("st-xgboost")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_DATA_FILE = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments and validate their values."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the RFE-Optuna-XGBoost water quality model "
            "for NDCI, NDTI and FUI."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Feature table (CSV) with one row per grid cell and year.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics, figures and model files.",
    )
    parser.add_argument(
        "--target",
        type=lambda value: value.upper(),
        choices=(*TARGETS, "ALL"),
        default="ALL",
        help="Water quality indicator to model, or ALL three in one run.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        help="Year held out for testing; all earlier years are used for training.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=15,
        help="Number of features kept by recursive feature elimination.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=0,
        help=(
            "0 = single chronological hold-out split for Optuna (paper setup); "
            "2 or more = expanding-window time-series cross-validation with "
            "that many folds."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for XGBoost, Optuna and the diagnostic figures.",
    )
    parser.add_argument(
        "--trim-low",
        type=float,
        default=0.01,
        help="Lower quantile used to remove target outliers from the training set.",
    )
    parser.add_argument(
        "--trim-high",
        type=float,
        default=0.99,
        help="Upper quantile used to remove target outliers from the training set.",
    )
    parser.add_argument(
        "--fui-offset",
        type=float,
        default=None,
        help=(
            "If given, convert FUI once with FUI = offset - FUI.  Use 0.6 for "
            "tables that store the raw hue instead of the paper convention."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of threads used by XGBoost and the k-nearest-neighbor search.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write tables and model files but no PNG figures.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log to <out>/run.log, not to the console.",
    )

    args = parser.parse_args(argv)

    if args.trials < 1:
        parser.error("--trials must be a positive integer")
    if args.n_features < 1:
        parser.error("--n-features must be a positive integer")
    if args.cv_splits == 1:
        parser.error("--cv-splits must be 0 (hold-out) or at least 2 (folds)")
    if not 0.0 <= args.trim_low < args.trim_high <= 1.0:
        parser.error("--trim-low must be >= 0 and smaller than --trim-high <= 1")
    return args


def setup_logging(output_dir: Path, quiet: bool = False) -> None:
    """Send the run log to <out>/run.log and, unless quiet, to stdout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        output_dir / "run.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    if not quiet:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def package_versions() -> Dict[str, str]:
    """Collect the versions needed to reproduce a run."""
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "optuna": optuna.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
    }


def _json_default(value):
    """Serialise numpy and Path objects for json.dump."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Dict) -> None:
    """Write a UTF-8 JSON file with a stable key order."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def run_target(
    target: str,
    feature_table: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Train, tune, evaluate and report one water quality indicator."""
    started = time.time()
    target_dir = output_root / target
    target_dir.mkdir(parents=True, exist_ok=True)

    features = feature_names_for(target)
    missing = [column for column in features if column not in feature_table.columns]
    if missing:
        raise ValueError(
            f"Feature table lacks predictor(s) for {target}: " + ", ".join(missing)
        )

    LOGGER.info("=" * 70)
    LOGGER.info("Target: %s  (%d predictors before RFE)", target, len(features))

    train_frame, test_frame = split_train_test(
        feature_table,
        target,
        args.test_year,
        args.trim_low,
        args.trim_high,
    )

    y_train = train_frame[target].to_numpy(dtype=np.float64)
    y_test = test_frame[target].to_numpy(dtype=np.float64)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_frame[features])
    X_test = scaler.transform(test_frame[features])

    # Conventional baseline -------------------------------------------------
    # The baseline deliberately uses the XGBoost library defaults: no tuned
    # hyperparameters are passed, so the comparison isolates the contribution
    # of the RFE-Optuna framework.
    LOGGER.info("Baseline XGBoost (library defaults) ...")
    baseline = xgb.XGBRegressor()
    baseline.fit(X_train, y_train)
    baseline_train = baseline.predict(X_train)
    baseline_test = baseline.predict(X_test)

    metrics = [
        regression_metrics(y_train, baseline_train, "Baseline_Train"),
        regression_metrics(y_test, baseline_test, "Baseline_Test"),
    ]
    LOGGER.info(
        "  baseline R2: train %.4f | test %.4f",
        metrics[0]["R2"],
        metrics[1]["R2"],
    )

    # RFE -------------------------------------------------------------------
    support = select_features(
        X_train, y_train, args.n_features, args.seed, args.n_jobs
    )
    selected = [name for name, keep in zip(features, support) if keep]
    LOGGER.info("  selected features (%d): %s", len(selected), ", ".join(selected))

    X_train_selected = X_train[:, support]
    X_test_selected = X_test[:, support]

    # Optuna ----------------------------------------------------------------
   
    study = tune_hyperparameters(
        X_train_selected,
        y_train,
        args.trials,
        args.seed + TARGETS.index(target),
        args.cv_splits,
        args.n_jobs,
    )

    model = train_final_model(
        X_train_selected,
        y_train,
        X_test_selected,
        y_test,
        study.best_params,
        args.seed,
        args.n_jobs,
    )
    framework_train = model.predict(X_train_selected)
    framework_test = model.predict(X_test_selected)

    metrics.extend(
        [
            regression_metrics(y_train, framework_train, "Framework_Train"),
            regression_metrics(y_test, framework_test, "Framework_Test"),
        ]
    )

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(target_dir / "Metrics_Comparison.csv", index=False)
    LOGGER.info(
        "Model performance for %s:\n%s",
        target,
        metrics_frame.to_string(index=False),
    )

    # Diagnostics -----------------------------------------------------------
    if not args.skip_plots:
        plot_diagnostics(
            target=target,
            model=model,
            feature_names=selected,
            y_train=y_train,
            yp_train=framework_train,
            y_test=y_test,
            yp_test=framework_test,
            study=study,
            train_frame=train_frame,
            test_frame=test_frame,
            test_year=args.test_year,
            output_dir=target_dir,
            rng=rng,
        )

    # artifacts -------------------------------------------------------------
    joblib.dump(model, target_dir / "Final_Model.pkl")
    joblib.dump(scaler, target_dir / "Scaler.pkl")
    joblib.dump(selected, target_dir / "Selected_Features.pkl")

    pd.DataFrame(
        {
            "Feature": selected,
            "Importance": np.asarray(model.feature_importances_, dtype=np.float64),
        }
    ).sort_values("Importance", ascending=False).to_csv(
        target_dir / "Feature_Importance.csv", index=False
    )

    pd.DataFrame(
        {
            "longitude": test_frame["longitude"].to_numpy(),
            "latitude": test_frame["latitude"].to_numpy(),
            "year": test_frame["year"].to_numpy(),
            "Observed": y_test,
            "Predicted": framework_test,
            "Residual": y_test - framework_test,
        }
    ).to_csv(target_dir / "Predictions_Test_Year.csv", index=False)

    trials_frame = pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": str(trial.state),
                "score": trial.value,
                **trial.params,
            }
            for trial in study.trials
        ]
    )
    trials_frame.to_csv(target_dir / "Optuna_Trials.csv", index=False)

    write_json(
        target_dir / "best_hyperparameters.json",
        {
            "target": target,
            "selected_features": selected,
            "best_hyperparameters": study.best_params,
            "best_score": study.best_value,
            "n_trials": args.trials,
            "cv_splits": args.cv_splits,
        },
    )

    write_json(
        target_dir / "run_metadata.json",
        {
            "target": target,
            "data_file": str(args.data),
            "test_year": args.test_year,
            "train_years": [
                int(train_frame["year"].min()),
                int(train_frame["year"].max()),
            ],
            "n_train": int(len(train_frame)),
            "n_test": int(len(test_frame)),
            "predictors": features,
            "selected_features": selected,
            "best_hyperparameters": study.best_params,
            "best_score": study.best_value,
            "metrics": metrics,
            "seed": args.seed,
            "trials": args.trials,
            "cv_splits": args.cv_splits,
            "n_jobs": args.n_jobs,
            "fui_offset": args.fui_offset,
            "versions": package_versions(),
            "platform": platform.platform(),
        },
    )

    baseline_r2 = float(metrics[1]["R2"])
    framework_r2 = float(metrics[3]["R2"])
    LOGGER.info(
        "%s finished in %.1f min | baseline R2 %.4f | framework R2 %.4f",
        target,
        (time.time() - started) / 60.0,
        baseline_r2,
        framework_r2,
    )

    return {
        "Target": target,
        "Baseline_R2": baseline_r2,
        "Framework_R2": framework_r2,
        "Baseline_RMSE": float(metrics[1]["RMSE"]),
        "Framework_RMSE": float(metrics[3]["RMSE"]),
        "N_train": int(len(y_train)),
        "N_test": int(len(y_test)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete workflow and return a process exit code."""
    args = parse_args(argv)
    configure_matplotlib()

    output_root = Path(args.out).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root, quiet=args.quiet)

    started = time.time()
    LOGGER.info("ST-XGBoost training run")
    LOGGER.info("Command line: %s", " ".join(sys.argv))
    LOGGER.info("Working directory: %s", Path.cwd())
    LOGGER.info("Data file: %s", args.data)
    LOGGER.info("Output directory: %s", output_root)
    LOGGER.info("Versions: %s", package_versions())

    frame = load_dataset(Path(args.data).expanduser(), args.fui_offset)
    frame = add_spatial_background(frame, k_neighbors=K_BACKGROUND)
    feature_table = build_feature_table(frame)
    del frame

    targets = TARGETS if args.target == "ALL" else (args.target,)
    rng = np.random.default_rng(args.seed)

    summary_rows = []
    for target in targets:
        summary_rows.append(
            run_target(target, feature_table, args, output_root, rng)
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "R2_Summary.csv", index=False)

    if not args.skip_plots:
        plot_r2_comparison(summary, output_root)

    LOGGER.info("=" * 70)
    LOGGER.info("Final summary (held-out test year %d):", args.test_year)
    for row in summary_rows:
        LOGGER.info(
            "  %-4s baseline R2 %.4f | framework R2 %.4f",
            row["Target"],
            row["Baseline_R2"],
            row["Framework_R2"],
        )
    LOGGER.info(
        "Run finished in %.1f min; outputs written to %s",
        (time.time() - started) / 60.0,
        output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
