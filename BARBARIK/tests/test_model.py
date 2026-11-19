"""
BARBARIK - model tests
======================

Sanity checks that a trained model can be loaded and evaluated, and that the
model module's public functions behave as expected.

Run with:  python -m pytest tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import load_model, save_model, train_model, evaluate_model  # noqa: E402
from preprocessing import (  # noqa: E402
    load_dataset,
    split_data,
    build_preprocessing_pipeline,
)


def _loaded():
    model, pipeline = load_model("models/")
    return model, pipeline


def test_model_and_pipeline_load():
    model, pipeline = _loaded()
    assert model is not None
    assert pipeline is not None
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == ["SIDE_A_ADVANTAGE", "SIDE_B_ADVANTAGE", "STALEMATE"]


def test_evaluate_metrics_keys_and_types():
    model, _ = _loaded()
    df = load_dataset("data/barbarik_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_preprocessing_pipeline()
    X_train_pp = pipeline.fit_transform(X_train, y_train)
    X_test_pp = pipeline.transform(X_test)

    # Train a fresh model on preprocessed data to keep the test self-contained.
    model = train_model(X_train_pp, y_train)
    metrics = evaluate_model(model, X_test_pp, y_test)

    for key in (
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "classification_report", "confusion_matrix", "roc_auc",
    ):
        assert key in metrics, f"missing key {key}"

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    assert metrics["confusion_matrix"].shape[0] == len(model.classes_)
    assert isinstance(metrics["classification_report"], str)


def test_save_then_load_roundtrip(tmp_path):
    df = load_dataset("data/barbarik_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_preprocessing_pipeline()
    X_train_pp = pipeline.fit_transform(X_train, y_train)

    model = train_model(X_train_pp, y_train)
    save_model(model, pipeline, model_dir=str(tmp_path))
    loaded_model, loaded_pipe = load_model(model_dir=str(tmp_path))

    assert loaded_model is not None
    assert loaded_pipe is not None