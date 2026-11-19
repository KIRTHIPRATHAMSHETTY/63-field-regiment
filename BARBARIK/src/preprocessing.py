"""
BARBARIK - Preprocessing Pipeline
=================================

Builds a reusable preprocessing :class:`sklearn.pipeline.Pipeline` for the
fictional BARBARIK dataset and provides loading / splitting helpers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns present in the generated CSV.
FEATURE_COLUMNS = [
    "side_a_force_strength",
    "side_b_force_strength",
    "side_a_mobility",
    "side_b_mobility",
    "side_a_logistics",
    "side_b_logistics",
    "side_a_intelligence",
    "side_b_intelligence",
    "side_a_communication",
    "side_b_communication",
    "side_a_readiness",
    "side_b_readiness",
    "side_a_morale",
    "side_b_morale",
    "side_a_training",
    "side_b_training",
    "side_a_supply_distance",
    "side_b_supply_distance",
    "side_a_surprise",
    "side_b_surprise",
    "terrain_type",
    "weather",
    "visibility",
    "time_of_day",
    "environmental_difficulty",
    "electronic_warfare_level",
    "objective_type",
]

NUMERIC_COLUMNS = [
    "side_a_force_strength",
    "side_b_force_strength",
    "side_a_mobility",
    "side_b_mobility",
    "side_a_logistics",
    "side_b_logistics",
    "side_a_intelligence",
    "side_b_intelligence",
    "side_a_communication",
    "side_b_communication",
    "side_a_readiness",
    "side_b_readiness",
    "side_a_morale",
    "side_b_morale",
    "side_a_training",
    "side_b_training",
    "side_a_supply_distance",
    "side_b_supply_distance",
    "side_a_surprise",
    "side_b_surprise",
    "environmental_difficulty",
    "electronic_warfare_level",
]

CATEGORICAL_COLUMNS = [
    "terrain_type",
    "weather",
    "visibility",
    "time_of_day",
    "objective_type",
]

TARGET_COLUMN = "outcome"


def build_preprocessing_pipeline() -> Pipeline:
    """Build a ColumnTransformer-based preprocessing pipeline.

    Numeric features are standardised (StandardScaler).  Standardisation is a
    good default for distance- and gradient-based learners and costs nothing
    for tree models, so we keep one shared pipeline.  Categorical features are
    one-hot encoded with ``handle_unknown="ignore"`` so unseen levels during
    scoring do not crash the model.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    return pipeline


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset and validate it.

    Parameters
    ----------
    path : str
        Path to the generated CSV.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file is empty or contains missing values.
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset at {path!r} is empty.")

    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in columns: {missing}")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split on the target column.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test