"""
BARBARIK - Model training / evaluation / persistence
====================================================

Trains a Random Forest classifier on the fictional BARBARIK dataset and
provides helpers for evaluating, saving, loading and predicting.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_FILENAME = "random_forest_model.joblib"
PIPELINE_FILENAME = "preprocessing_pipeline.joblib"

OUTCOME_CLASSES = ["SIDE_A_ADVANTAGE", "SIDE_B_ADVANTAGE", "STALEMATE"]


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier on preprocessed features."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute classification metrics for a fitted model.

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        classification_report (str), confusion_matrix (np.ndarray),
        roc_auc (float or None)
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC (multi-class). Requires predict_proba. Falls back to None if the
    # model does not expose probabilities (e.g. some classifiers).
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="macro",
            )
        except (ValueError, NotImplementedError):
            roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
    }


def save_model(model, preprocessing_pipeline, model_dir: str = "models/") -> None:
    """Persist the fitted model and its preprocessing pipeline to disk."""
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / MODEL_FILENAME)
    joblib.dump(preprocessing_pipeline, out_dir / PIPELINE_FILENAME)


def load_model(model_dir: str = "models/"):
    """Load the model + preprocessing pipeline saved by :func:`save_model`."""
    out_dir = Path(model_dir)
    model = joblib.load(out_dir / MODEL_FILENAME)
    pipeline = joblib.load(out_dir / PIPELINE_FILENAME)
    return model, pipeline


def predict_outcome(model, pipeline, scenario_dict: dict) -> dict:
    """Predict a single scenario outcome from a raw input dict.

    Parameters
    ----------
    scenario_dict : dict
        Must contain the raw feature keys (numerics + categoricals) exactly as
        they appear in the dataset (minus scenario_id / outcome).

    Returns
    -------
    dict with:
        predicted_class, probabilities (dict class -> probability)
    """
    # Accept any extra keys (e.g. scenario_id) by only selecting the feature
    # columns known to the pipeline.
    row = pd.DataFrame([scenario_dict])
    X = pipeline.transform(row)

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    probs = {cls: float(p) for cls, p in zip(classes, proba)}

    predicted_class = classes[int(np.argmax(proba))]

    return {
        "predicted_class": predicted_class,
        "probabilities": probs,
    }