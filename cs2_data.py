"""
cs2_data.py
===========

Data pipeline for the CS2 match-prediction project.

This module takes the raw Counter-Strike 2 match data (one row per match, with
detailed per-player statistics) and turns it into the clean, per-team feature
table that the models are trained on.

The flow is:

    raw HLTV data (Kaggle)
        -> clean_raw_data()        # tidy column names, fix %, handle missing values
        -> build_team_features()   # collapse 5 players/team into team-level stats
        -> team_summary_statistics.csv   (the table every model uses)

Small helpers (load_features / split_xy) cover the loading and train/test
splitting used by the modelling notebook.

Nothing here needs a GPU or internet except download_dataset().
"""

from __future__ import annotations

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constants shared across the whole project
# ---------------------------------------------------------------------------

# Columns that identify a match but are NOT statistics the models learn from.
# ('map' is sometimes kept as a categorical feature for LightGBM - see notebook.)
ID_COLUMNS = ["matchID", "team1", "team2", "map", "team1_win",
              "T1_mapwinrate", "T2_mapwinrate"]

# The prediction target: 1 if team1 won the match, 0 otherwise.
TARGET = "team1_win"

# Per-player columns look like  "T1_player0_Total_kills" / "T2_player4_K/D_Ratio".
# This pattern captures the statistic name (the part after the player number).
PLAYER_STAT_PATTERN = r"T[12]_player[0-4]_(.+)"

# The five team-level summaries we compute from each squad's five players.
AGGREGATES = ["mean", "sum", "std", "min", "max"]

# Default output filenames (kept flat in the project folder).
CLEAN_WITH_NAN_CSV = "CS2_CLEANDATA_with_nan.csv"
CLEAN_ZERO_FILLED_CSV = "CS2_CLEANDATA_zero_filled.csv"
CLEAN_DROPPED_CSV = "CS2_CLEANDATA_all_nan_dropped.csv"
TEAM_FEATURES_CSV = "team_summary_statistics.csv"


# ---------------------------------------------------------------------------
# Step 0 - get the raw data
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """Download the HLTV CS2 dataset from Kaggle and return the local folder path.

    Requires the `kagglehub` package and Kaggle credentials configured on the
    machine. You only need to run this once; afterwards the cleaned CSVs are
    cached in the project folder.
    """
    import kagglehub  # imported lazily so the rest of the module works offline

    path = kagglehub.dataset_download(
        "victorpicinin/counter-strike-2-hltv-match-data"
    )
    print("Dataset files:", os.listdir(path))
    return path


# ---------------------------------------------------------------------------
# Step 1 - clean the raw match data
# ---------------------------------------------------------------------------

def _clean_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any column containing '%' values (e.g. '63%') into decimals (0.63)."""
    percentage_columns = [
        col for col in df.columns
        if df[col].astype(str).str.contains("%").any()
    ]
    for col in percentage_columns:
        df[col] = (
            df[col].astype(str)
                   .str.replace("%", "", regex=False)
                   .astype(float)
            / 100
        )
    return df


def clean_raw_data(raw_csv_path: str,
                   delimiter: str = ";",
                   min_non_null_fraction: float = 0.9,
                   save: bool = True) -> dict[str, pd.DataFrame]:
    """Clean the raw match CSV and return three missing-value variants.

    Cleaning steps:
      1. Replace spaces in column names with underscores.
      2. Drop columns that are mostly empty (less than `min_non_null_fraction`
         non-missing values).
      3. Turn percentage strings into decimals.
      4. Drop rows whose map is 'TBA' or 'Default' (not real matches).

    Because different models treat missing values differently, three versions
    are produced:
      * "with_nan"        - missing win-rates left as NaN (e.g. TabPFN tolerates this).
      * "zero_filled"     - missing T1/T2 map win-rates filled with 0.
      * "all_nan_dropped" - every row with any missing value removed.

    Returns a dict with keys: 'with_nan', 'zero_filled', 'all_nan_dropped'.
    """
    df = pd.read_csv(raw_csv_path, delimiter=delimiter)

    # 1. tidy column names
    df.columns = df.columns.str.replace(" ", "_")

    # 2. drop columns that are mostly missing
    threshold = len(df) * min_non_null_fraction  # keep cols with >= 90% non-NaN
    df = df.dropna(axis=1, thresh=threshold)

    # 3. percentages -> decimals
    df = _clean_percentage_columns(df)

    # 4. drop placeholder matches
    df = df[~df["map"].isin(["TBA", "Default"])]

    # --- three missing-value variants ---
    df_with_nan = df.copy()

    df_zero_filled = df.copy()
    df_zero_filled["T1_mapwinrate"] = df_zero_filled["T1_mapwinrate"].fillna(0.0)
    df_zero_filled["T2_mapwinrate"] = df_zero_filled["T2_mapwinrate"].fillna(0.0)

    df_all_nan_dropped = df.copy()
    df_all_nan_dropped.dropna(inplace=True)

    if save:
        df_with_nan.to_csv(CLEAN_WITH_NAN_CSV, index=False)
        df_zero_filled.to_csv(CLEAN_ZERO_FILLED_CSV, index=False)
        df_all_nan_dropped.to_csv(CLEAN_DROPPED_CSV, index=False)

    return {
        "with_nan": df_with_nan,
        "zero_filled": df_zero_filled,
        "all_nan_dropped": df_all_nan_dropped,
    }


# ---------------------------------------------------------------------------
# Step 2 - build per-team features
# ---------------------------------------------------------------------------

def build_team_features(df: pd.DataFrame,
                        save_path: str | None = TEAM_FEATURES_CSV) -> pd.DataFrame:
    """Collapse the 5 players on each team into team-level summary statistics.

    For every per-player statistic (e.g. 'Total_kills') and every team (T1/T2),
    this computes the mean / sum / std / min / max across that team's 5 players,
    producing columns like 'T1_Total_kills_mean'. The match-identifier columns
    and target are carried through unchanged.

    Works on any frame with columns matching the per-player naming pattern.
    """
    # Find every distinct statistic name present in the per-player columns.
    stat_types = set()
    for col in df.columns:
        match = re.match(PLAYER_STAT_PATTERN, col)
        if match:
            stat_types.add(match.group(1))

    # sorted() keeps the output column order deterministic across runs.
    aggregated_blocks = []
    for team in ["T1", "T2"]:
        for stat in sorted(stat_types):
            stat_cols = [f"{team}_player{i}_{stat}" for i in range(5)
                         if f"{team}_player{i}_{stat}" in df.columns]
            if not stat_cols:
                continue

            # Make sure the values are numeric before aggregating.
            stat_data = (df[stat_cols]
                         .replace("%", "", regex=True)
                         .apply(pd.to_numeric, errors="coerce"))

            block = pd.DataFrame()
            for agg in AGGREGATES:
                block[f"{team}_{stat}_{agg}"] = getattr(stat_data, agg)(axis=1)
            aggregated_blocks.append(block)

    aggregated_df = pd.concat(aggregated_blocks, axis=1)

    # Keep the identifier columns alongside the new team-level features.
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    final_df = pd.concat([df[id_cols].reset_index(drop=True),
                          aggregated_df.reset_index(drop=True)], axis=1)

    if save_path:
        final_df.to_csv(save_path, index=False)

    return final_df


# ---------------------------------------------------------------------------
# Step 3 - convenience loaders for the modelling notebooks
# ---------------------------------------------------------------------------

def load_features(path: str = TEAM_FEATURES_CSV) -> pd.DataFrame:
    """Load the team-level feature table that the models train on."""
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame,
             drop_cols: list[str] | None = None,
             keep_map: bool = False,
             test_size: float = 0.2,
             random_state: int = 42,
             stratify: bool = True):
    """Split a feature table into train/test feature matrices and target.

    drop_cols : identifier columns to remove before modelling. Defaults to
                everything in ID_COLUMNS except the target (and except 'map'
                when keep_map=True, e.g. for LightGBM's categorical handling).
    keep_map  : if True, 'map' is kept as a feature (and not dropped).

    Returns (X_train, X_test, y_train, y_test).
    """
    if drop_cols is None:
        drop_cols = [c for c in ["matchID", "team1", "team2", "map"]
                     if not (keep_map and c == "map")]

    X = df.drop(columns=drop_cols + [TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


# ---------------------------------------------------------------------------
# One-call pipeline: raw data -> ready-to-model feature table
# ---------------------------------------------------------------------------

def prepare_training_data(raw_csv_path: str) -> pd.DataFrame:
    """Run the full pipeline (clean -> features) and return the feature table.

    Uses the 'all_nan_dropped' variant (rows with missing values removed)
    for feature building.
    """
    cleaned = clean_raw_data(raw_csv_path)
    return build_team_features(cleaned["all_nan_dropped"])
