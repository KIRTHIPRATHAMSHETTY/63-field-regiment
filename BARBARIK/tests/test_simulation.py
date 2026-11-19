"""
Sanity tests for the BARBARIK simulation engine.

Run with:  python -m pytest tests/test_simulation.py -v
"""

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation_engine import simulate_scenario, TERRAIN_TYPES  # noqa: E402


def _base_config(**overrides) -> dict:
    """A balanced, mid-level scenario to use as a starting point."""
    config = {
        "side_a_force_strength": 50,
        "side_b_force_strength": 50,
        "side_a_mobility": 0.5,
        "side_b_mobility": 0.5,
        "side_a_logistics": 0.5,
        "side_b_logistics": 0.5,
        "side_a_intelligence": 0.5,
        "side_b_intelligence": 0.5,
        "side_a_communication": 0.5,
        "side_b_communication": 0.5,
        "side_a_readiness": 0.5,
        "side_b_readiness": 0.5,
        "terrain_type": "Open",
        "weather": "Clear",
        "visibility": "High",
        "time_of_day": "Day",
        "environmental_difficulty": 0.3,
        "simulation_duration": 5,
    }
    config.update(overrides)
    return config


def test_high_force_and_logistics_favours_side_a():
    """A dominant side A should win most of the time over 100 runs."""
    config = _base_config(
        side_a_force_strength=95,
        side_b_force_strength=15,
        side_a_logistics=0.9,
        side_b_logistics=0.2,
        side_a_intelligence=0.85,
        side_b_intelligence=0.3,
        side_a_mobility=0.8,
        side_b_mobility=0.3,
        side_a_readiness=0.85,
        side_b_readiness=0.3,
    )

    n_runs = 100
    wins = sum(
        simulate_scenario(config)["outcome"] == "SIDE_A_ADVANTAGE"
        for _ in range(n_runs)
    )

    # Allow a few stochastic flukes but expect a clear majority.
    assert wins >= 90, f"Side A advantage only in {wins}/{n_runs} runs"


def test_equal_parameters_give_mixed_outcomes():
    """Symmetric inputs must produce a mix, never 100% of one class."""
    config = _base_config()  # fully balanced

    n_runs = 100
    outcomes = [simulate_scenario(config)["outcome"] for _ in range(n_runs)]

    n_a = outcomes.count("SIDE_A_ADVANTAGE")
    n_b = outcomes.count("SIDE_B_ADVANTAGE")
    n_stalemate = outcomes.count("STALEMATE")

    assert n_a != 100 and n_b != 100 and n_stalemate != 100, (
        f"Expected a mix, got A={n_a}, B={n_b}, S={n_stalemate}"
    )
    # With symmetry the outcome distribution should be reasonably balanced.
    assert not (n_a > 80 or n_b > 80), (
        f"Distribution unexpectedly lopsided: A={n_a}, B={n_b}, S={n_stalemate}"
    )


def test_returned_keys_and_types():
    """The returned dict must have all keys with the correct types."""
    for terrain in TERRAIN_TYPES:
        config = _base_config(terrain_type=terrain)
        result = simulate_scenario(config)

        expected_keys = {
            "outcome",
            "duration",
            "simulation_score_a",
            "simulation_score_b",
            "resource_status_a",
            "resource_status_b",
        }
        assert set(result.keys()) == expected_keys, (
            f"Mismatched keys for terrain={terrain}: {set(result.keys())}"
        )

        assert result["outcome"] in {
            "SIDE_A_ADVANTAGE",
            "SIDE_B_ADVANTAGE",
            "STALEMATE",
        }
        assert isinstance(result["duration"], int)
        assert 1 <= result["duration"] <= 10

        assert isinstance(result["simulation_score_a"], (int, float))
        assert isinstance(result["simulation_score_b"], (int, float))
        assert np.isfinite(result["simulation_score_a"])
        assert np.isfinite(result["simulation_score_b"])

        assert result["resource_status_a"] in {"Low", "Medium", "High"}
        assert result["resource_status_b"] in {"Low", "Medium", "High"}