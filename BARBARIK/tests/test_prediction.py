"""
BARBARIK - prediction tests
===========================

Tests for the prediction helper: loading a scenario dict, transforming it
through the pipeline and producing a valid outcome prediction.

Run with:  python -m pytest tests/test_prediction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import load_model, predict_outcome  # noqa: E402


def _scenario() -> dict:
    """A realistic balanced scenario dict (all 18 feature keys)."""
    return {
        "side_a_force_strength": 60,
        "side_b_force_strength": 45,
        "side_a_mobility": 0.7,
        "side_b_mobility": 0.4,
        "side_a_logistics": 0.8,
        "side_b_logistics": 0.3,
        "side_a_intelligence": 0.7,
        "side_b_intelligence": 0.4,
        "side_a_communication": 0.8,
        "side_b_communication": 0.3,
        "side_a_readiness": 0.7,
        "side_b_readiness": 0.4,
        "terrain_type": "Open",
        "weather": "Clear",
        "visibility": "High",
        "time_of_day": "Day",
        "environmental_difficulty": 0.3,
        "simulation_duration": 5,
    }


def test_predict_one_row_returns_expected_keys():
    model, pipeline = load_model("models/")
    result = predict_outcome(model, pipeline, _scenario())

    assert "predicted_class" in result
    assert "probabilities" in result

    pred = result["predicted_class"]
    assert pred in {"SIDE_A_ADVANTAGE", "SIDE_B_ADVANTAGE", "STALEMATE"}

    probs = result["probabilities"]
    assert set(probs.keys()) == set(model.classes_)
    # probabilities should be non-negative and roughly sum to 1.
    assert all(p >= 0.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_probabilities_consistent_with_joint():
    model, pipeline = load_model("models/")
    result = predict_outcome(model, pipeline, _scenario())

    # The predicted class must be the one with max probability.
    probs = result["probabilities"]
    top_class = max(probs, key=probs.get)
    assert result["predicted_class"] == top_class


def test_predict_output_types():
    model, pipeline = load_model("models/")
    result = predict_outcome(model, pipeline, _scenario())
    assert isinstance(result["predicted_class"], str)
    assert isinstance(result["probabilities"], dict)
    for v in result["probabilities"].values():
        assert isinstance(v, float)