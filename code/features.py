"""Feature engineering and the temporal train/test split."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

LOGGER = logging.getLogger("st-xgboost")


TARGETS: Tuple[str, ...] = ("NDCI", "NDTI", "FUI")

GRID_COLUMNS: Tuple[str, ...] = ("longitude", "latitude", "year")

RAW_FEATURES: Tuple[str, ...] = (
    "built_1km",
    "crop_1km",
    "light_1km",
    "ndvi_1km",
    "rain_sum",
    "temp_avg",
    "dem",
    "slope_1km",
    "water_occurrence",
    "Relative_Depth",
)

CONTEXT_FEATURES: Tuple[str, ...] = ("Region_Rain", "Region_Crop")

INTERACTION_FEATURES: Tuple[str, ...] = (
    "Rain_Crop",
    "Rain_Slope",
    "Local_Pollution_Rain",
    "Delta_rain_sum",
    "Delta_crop_1km",
)

BASE_FEATURES: Tuple[str, ...] = (
    RAW_FEATURES + CONTEXT_FEATURES + INTERACTION_FEATURES
)

TARGET_SPECIFIC_FEATURES: Dict[str, Tuple[str, ...]] = {
    "NDCI": ("Temp_PrevNDCI",),
}

REQUIRED_COLUMNS: Tuple[str, ...] = GRID_COLUMNS + TARGETS + RAW_FEATURES

K_BACKGROUND = 50

K_SPATIAL_LAG = 5

BALL_TREE_LEAF_SIZE = 40


def load_dataset(path: Path, fui_offset: Optional[float]) -> pd.DataFrame:
    """
    Read and validate the feature table.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. "
            "Place the exported CSV there or pass --data /path/to/file.csv."
        )

    LOGGER.info("Loading feature table: %s", path)
    frame = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "The feature table is missing required column(s): " + ", ".join(missing)
        )

    if fui_offset is not None:
        LOGGER.info(
            "Converting FUI to the paper convention once: FUI = %.3f - FUI",
            fui_offset,
        )
        frame["FUI"] = float(fui_offset) - frame["FUI"]

    frame = frame.sort_values(
        ["year", "longitude", "latitude"], kind="mergesort"
    ).reset_index(drop=True)

    n_duplicates = int(
        frame.duplicated(subset=["longitude", "latitude", "year"]).sum()
    )
    if n_duplicates:
        raise ValueError(
            f"The feature table contains {n_duplicates:,} duplicated "
            "(longitude, latitude, year) record(s). The table must hold one "
            "row per 100 m grid cell and year."
        )

    LOGGER.info(
        "Loaded %s rows x %d columns (%d-%d)",
        f"{len(frame):,}",
        frame.shape[1],
        int(frame["year"].min()),
        int(frame["year"].max()),
    )
    return frame


def add_spatial_background(
    frame: pd.DataFrame,
    k_neighbors: int = K_BACKGROUND,
    leaf_size: int = BALL_TREE_LEAF_SIZE,
) -> pd.DataFrame:
    """
    Add the neighborhood-mean background of rainfall and cropland.
    """
    n_rows = len(frame)
    region_rain = np.empty(n_rows, dtype=np.float64)
    region_crop = np.empty(n_rows, dtype=np.float64)

    coordinates = frame[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    rain = frame["rain_sum"].to_numpy(dtype=np.float64)
    crop = frame["crop_1km"].to_numpy(dtype=np.float64)

    for year, positions in frame.groupby("year", sort=True).indices.items():
        positions = np.asarray(positions)
        k = min(k_neighbors, positions.size)
        tree = BallTree(coordinates[positions], leaf_size=leaf_size)
        _, neighbors = tree.query(coordinates[positions], k=k)
        neighbors = np.asarray(neighbors).reshape(positions.size, k)

        region_rain[positions] = rain[positions][neighbors].mean(axis=1)
        region_crop[positions] = crop[positions][neighbors].mean(axis=1)
        LOGGER.info("  spatial background %d: %s cells", year, f"{positions.size:,}")

    frame["Region_Rain"] = region_rain
    frame["Region_Crop"] = region_crop
    return frame


def add_spatial_lag(
    frame: pd.DataFrame,
    target: str,
    k_neighbors: int = K_SPATIAL_LAG,
    leaf_size: int = BALL_TREE_LEAF_SIZE,
) -> pd.DataFrame:
    """
    Add the previous-year mean target value of the k nearest cells.
    """
    years = np.sort(frame["year"].unique())
    coordinates = frame[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    values = frame[target].to_numpy(dtype=np.float64)
    year_values = frame["year"].to_numpy()
    spatial_lag = np.full(len(frame), np.nan, dtype=np.float64)

    for index, year in enumerate(years):
        current = np.flatnonzero(year_values == year)
        if index == 0:
            spatial_lag[current] = values[current].mean()
            continue

        previous = np.flatnonzero(year_values == years[index - 1])
        k = min(k_neighbors, previous.size)
        tree = BallTree(coordinates[previous], leaf_size=leaf_size)
        _, neighbors = tree.query(coordinates[current], k=k)
        neighbors = np.asarray(neighbors).reshape(current.size, k)
        spatial_lag[current] = values[previous][neighbors].mean(axis=1)

    frame[f"SpatialLag_{target}"] = spatial_lag
    return frame


def build_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, interaction and difference features.

    Features created
    ----------------
    Prev_<target>          previous-year value at the same cell
    SpatialLag_<target>    previous-year mean of the k nearest cells
    Rain_Crop              rain_sum x crop_1km
    Rain_Slope             rain_sum x slope_1km
    Local_Pollution_Rain   rain_sum - Region_Rain (local deviation from background)
    Temp_PrevNDCI          temp_avg x Prev_NDCI (NDCI model only)
    Delta_rain_sum         year-on-year rainfall difference at the same cell
    Delta_crop_1km         year-on-year cropland difference at the same cell
    """
    LOGGER.info("Building lagged and interaction features ...")

    for target in TARGETS:
        frame[f"Prev_{target}"] = frame.groupby(
            ["longitude", "latitude"], sort=False
        )[target].shift(1)

    for target in TARGETS:
        frame = add_spatial_lag(frame, target)

    frame["Rain_Crop"] = frame["rain_sum"] * frame["crop_1km"]
    frame["Rain_Slope"] = frame["rain_sum"] * frame["slope_1km"]
    frame["Local_Pollution_Rain"] = frame["rain_sum"] - frame["Region_Rain"]
    frame["Temp_PrevNDCI"] = frame["temp_avg"] * frame["Prev_NDCI"]

    for column in ("rain_sum", "crop_1km"):
        frame[f"Delta_{column}"] = frame.groupby(
            ["longitude", "latitude"], sort=False
        )[column].diff()

    n_before = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            "All rows were removed because the lagged features are undefined. "
            "Each grid cell must be identified by exactly the same longitude "
            "and latitude values in every year."
        )
    LOGGER.info(
        "Dropped %s row(s) with missing lagged values (first year of each cell); "
        "%s rows remain",
        f"{n_before - len(frame):,}",
        f"{len(frame):,}",
    )
    return frame


def feature_names_for(target: str) -> list:
    """Return the predictor list of one water quality indicator."""
    features = list(BASE_FEATURES)
    features.append(f"Prev_{target}")
    features.append(f"SpatialLag_{target}")
    features.extend(TARGET_SPECIFIC_FEATURES.get(target, ()))
    return features


def split_train_test(
    frame: pd.DataFrame,
    target: str,
    test_year: int,
    trim_low: float,
    trim_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the table by year and trim target outliers from the training set."""
    train = frame.loc[frame["year"] < test_year].copy()
    test = frame.loc[frame["year"] == test_year].copy()

    if train.empty:
        raise ValueError(
            f"No training rows before {test_year}; check --test-year and the "
            "year column of the feature table."
        )
    if test.empty:
        raise ValueError(
            f"No rows for the test year {test_year}; check --test-year and the "
            "year column of the feature table."
        )

    lower = float(train[target].quantile(trim_low))
    upper = float(train[target].quantile(trim_high))
    keep = (train[target] >= lower) & (train[target] <= upper)
    n_removed = int((~keep).sum())
    train = train.loc[keep].reset_index(drop=True)

    LOGGER.info(
        "Train: %s rows (%d-%d, target trimmed to [%.5f, %.5f], %s row(s) removed)",
        f"{len(train):,}",
        int(train["year"].min()),
        int(train["year"].max()),
        lower,
        upper,
        f"{n_removed:,}",
    )
    LOGGER.info("Test:  %s rows (year %d)", f"{len(test):,}", test_year)
    return train, test
