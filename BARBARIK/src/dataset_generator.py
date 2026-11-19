"""
BARBARIK dataset generator.

Produces a synthetic training dataset by sampling scenario inputs and running
the simulation engine for each generated row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation_engine import (
    OBJECTIVE_TYPES,
    TERRAIN_TYPES,
    TIME_TYPES,
    VISIBILITY_TYPES,
    WEATHER_TYPES,
    _core_simulate,
)

OUTPUT_COLUMNS = [
    "scenario_id",
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
    "outcome",
]


def generate_dataset(
    n_scenarios: int = 20000,
    output_path: str = "data/barbarik_dataset.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate the BARBARIK synthetic dataset and save it as CSV."""
    rng = np.random.default_rng(random_state)
    sampled = _sample_numeric_block(n_scenarios, rng)

    terrain = rng.choice(TERRAIN_TYPES, size=n_scenarios)
    weather = rng.choice(WEATHER_TYPES, size=n_scenarios)
    visibility = rng.choice(VISIBILITY_TYPES, size=n_scenarios)
    time_of_day = rng.choice(TIME_TYPES, size=n_scenarios)
    objective_type = rng.choice(OBJECTIVE_TYPES, size=n_scenarios)

    rows: list[list[object]] = []
    for i in range(n_scenarios):
        config = {
            "side_a_force_strength": int(round(sampled["force_a"][i])),
            "side_b_force_strength": int(round(sampled["force_b"][i])),
            "side_a_mobility": float(sampled["mobility_a"][i]),
            "side_b_mobility": float(sampled["mobility_b"][i]),
            "side_a_logistics": float(sampled["logistics_a"][i]),
            "side_b_logistics": float(sampled["logistics_b"][i]),
            "side_a_intelligence": float(sampled["intelligence_a"][i]),
            "side_b_intelligence": float(sampled["intelligence_b"][i]),
            "side_a_communication": float(sampled["communication_a"][i]),
            "side_b_communication": float(sampled["communication_b"][i]),
            "side_a_readiness": float(sampled["readiness_a"][i]),
            "side_b_readiness": float(sampled["readiness_b"][i]),
            "side_a_morale": float(sampled["morale_a"][i]),
            "side_b_morale": float(sampled["morale_b"][i]),
            "side_a_training": float(sampled["training_a"][i]),
            "side_b_training": float(sampled["training_b"][i]),
            "side_a_supply_distance": float(sampled["supply_distance_a"][i]),
            "side_b_supply_distance": float(sampled["supply_distance_b"][i]),
            "side_a_surprise": float(sampled["surprise_a"][i]),
            "side_b_surprise": float(sampled["surprise_b"][i]),
            "terrain_type": str(terrain[i]),
            "weather": str(weather[i]),
            "visibility": str(visibility[i]),
            "time_of_day": str(time_of_day[i]),
            "environmental_difficulty": float(sampled["difficulty"][i]),
            "electronic_warfare_level": float(sampled["ew_level"][i]),
            "objective_type": str(objective_type[i]),
        }
        result = _core_simulate(config, rng)
        rows.append(
            [
                f"SCN-{i + 1:05d}",
                config["side_a_force_strength"],
                config["side_b_force_strength"],
                config["side_a_mobility"],
                config["side_b_mobility"],
                config["side_a_logistics"],
                config["side_b_logistics"],
                config["side_a_intelligence"],
                config["side_b_intelligence"],
                config["side_a_communication"],
                config["side_b_communication"],
                config["side_a_readiness"],
                config["side_b_readiness"],
                config["side_a_morale"],
                config["side_b_morale"],
                config["side_a_training"],
                config["side_b_training"],
                config["side_a_supply_distance"],
                config["side_b_supply_distance"],
                config["side_a_surprise"],
                config["side_b_surprise"],
                config["terrain_type"],
                config["weather"],
                config["visibility"],
                config["time_of_day"],
                config["environmental_difficulty"],
                config["electronic_warfare_level"],
                config["objective_type"],
                result["outcome"],
            ]
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def _sample_numeric_block(n_scenarios: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    bins = np.arange(n_scenarios)

    def lhs_unit() -> np.ndarray:
        perm = rng.permutation(bins)
        return (perm + rng.random(n_scenarios)) / n_scenarios

    return {
        "force_a": 10 + lhs_unit() * 90,
        "force_b": 10 + lhs_unit() * 90,
        "mobility_a": lhs_unit(),
        "mobility_b": lhs_unit(),
        "logistics_a": lhs_unit(),
        "logistics_b": lhs_unit(),
        "intelligence_a": lhs_unit(),
        "intelligence_b": lhs_unit(),
        "communication_a": lhs_unit(),
        "communication_b": lhs_unit(),
        "readiness_a": lhs_unit(),
        "readiness_b": lhs_unit(),
        "morale_a": lhs_unit(),
        "morale_b": lhs_unit(),
        "training_a": lhs_unit(),
        "training_b": lhs_unit(),
        "supply_distance_a": lhs_unit() * 500,
        "supply_distance_b": lhs_unit() * 500,
        "surprise_a": lhs_unit(),
        "surprise_b": lhs_unit(),
        "difficulty": lhs_unit(),
        "ew_level": lhs_unit(),
    }

