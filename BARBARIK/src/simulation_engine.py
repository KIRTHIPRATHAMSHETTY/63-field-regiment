"""
BARBARIK simulation engine.

Implements a fictional scenario simulator for academic ML experiments.
The scoring follows the project spec exactly: weighted force attributes,
environment modifiers, two nonlinear interactions, and Gaussian noise.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TERRAIN_TYPES = ["Open", "Hilly", "Forest", "Desert", "Urban", "Mixed"]
WEATHER_TYPES = ["Clear", "Cloudy", "Rain", "Fog", "Extreme"]
VISIBILITY_TYPES = ["High", "Medium", "Low"]
TIME_TYPES = ["Day", "Night"]
OBJECTIVE_TYPES = ["Attack", "Defend", "Delay", "Withdraw"]


def simulate_scenario(config: dict[str, Any]) -> dict[str, Any]:
    """Simulate a single fictional scenario and return outcome metrics."""
    rng = np.random.default_rng()
    return _core_simulate(config, rng)


def _core_simulate(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    side_a = _extract_side(config, "a")
    side_b = _extract_side(config, "b")
    terrain = config["terrain_type"]
    weather = config["weather"]
    visibility = config["visibility"]
    time_of_day = config["time_of_day"]
    environmental_difficulty = float(config["environmental_difficulty"])
    electronic_warfare_level = float(config.get("electronic_warfare_level", 0.0))
    objective_type = config.get("objective_type", "Attack")

    _apply_terrain(side_a, terrain)
    _apply_terrain(side_b, terrain)
    _apply_weather(side_a, side_b, weather)
    _apply_visibility(side_a, visibility)
    _apply_visibility(side_b, visibility)
    _apply_time_of_day(side_a, time_of_day)
    _apply_time_of_day(side_b, time_of_day)

    score_a = _score_side(side_a, terrain, environmental_difficulty, electronic_warfare_level)
    score_b = _score_side(side_b, terrain, environmental_difficulty, electronic_warfare_level)

    score_a += float(rng.normal(0.0, 0.05))
    score_b += float(rng.normal(0.0, 0.05))

    # Apply objective type modifiers
    score_a, score_b = _apply_objective_modifiers(score_a, score_b, objective_type, terrain)

    # Apply surprise bonus
    if side_a["surprise"] > 0.7:
        score_a += 0.15
    if side_b["surprise"] > 0.7:
        score_b += 0.15

    differential = score_a - score_b
    if differential > 0.15:
        outcome = "SIDE_A_ADVANTAGE"
    elif differential < -0.15:
        outcome = "SIDE_B_ADVANTAGE"
    else:
        outcome = "STALEMATE"

    duration = _estimate_duration(differential, environmental_difficulty, terrain, weather, objective_type)

    return {
        "outcome": outcome,
        "duration": duration,
        "simulation_score_a": float(score_a),
        "simulation_score_b": float(score_b),
        "resource_status_a": _resource_status(
            side_a["logistics"], side_a["readiness"], environmental_difficulty
        ),
        "resource_status_b": _resource_status(
            side_b["logistics"], side_b["readiness"], environmental_difficulty
        ),
    }


def _extract_side(config: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "force": float(config[f"side_{prefix}_force_strength"]) / 100.0,
        "mobility": float(config[f"side_{prefix}_mobility"]),
        "logistics": float(config[f"side_{prefix}_logistics"]),
        "intelligence": float(config[f"side_{prefix}_intelligence"]),
        "communication": float(config[f"side_{prefix}_communication"]),
        "readiness": float(config[f"side_{prefix}_readiness"]),
        "morale": float(config[f"side_{prefix}_morale"]),
        "training": float(config[f"side_{prefix}_training"]),
        "supply_distance": float(config[f"side_{prefix}_supply_distance"]),
        "surprise": float(config[f"side_{prefix}_surprise"]),
    }


def _apply_terrain(side: dict[str, float], terrain: str) -> None:
    if terrain == "Open":
        side["mobility"] *= 1.3
        side["readiness"] *= 0.9
    elif terrain == "Hilly":
        side["readiness"] *= 1.3
        side["mobility"] *= 0.8
    elif terrain == "Forest":
        side["intelligence"] *= 1.2
        side["mobility"] *= 0.85
    elif terrain == "Desert":
        side["logistics"] *= 1.3
        side["mobility"] *= 0.9
    elif terrain == "Urban":
        side["readiness"] *= 1.15
        side["communication"] *= 1.15


def _apply_weather(side_a: dict[str, float], side_b: dict[str, float], weather: str) -> None:
    if weather == "Cloudy":
        for key in side_a:
            side_a[key] *= 0.95
            side_b[key] *= 0.95
    elif weather == "Rain":
        for key in ("logistics", "mobility"):
            side_a[key] *= 0.85
            side_b[key] *= 0.85
    elif weather == "Fog":
        if side_a["intelligence"] >= side_b["intelligence"]:
            side_a["intelligence"] *= 1.25
            side_b["intelligence"] *= 0.75
        else:
            side_b["intelligence"] *= 1.25
            side_a["intelligence"] *= 0.75
    elif weather == "Extreme":
        for key in side_a:
            side_a[key] *= 0.75
            side_b[key] *= 0.75


def _apply_visibility(side: dict[str, float], visibility: str) -> None:
    if visibility == "Medium":
        side["intelligence"] *= 0.95
    elif visibility == "Low":
        side["intelligence"] *= 0.85


def _apply_time_of_day(side: dict[str, float], time_of_day: str) -> None:
    if time_of_day == "Night":
        side["intelligence"] *= 1.15
        side["communication"] *= 0.9


def _score_side(
    side: dict[str, float],
    terrain: str,
    environmental_difficulty: float,
    electronic_warfare_level: float = 0.0,
) -> float:
    mobility_effective = side["mobility"] * (0.6 + 0.4 * _terrain_mobility_weight(terrain))
    
    # Apply electronic warfare to communication
    effective_communication = side["communication"] * (1 - electronic_warfare_level * 0.6)
    intel_effective = side["intelligence"] * (0.6 + 0.4 * effective_communication)

    # Apply morale modifier to readiness
    effective_readiness = side["readiness"] * (0.7 + 0.3 * side["morale"])

    # Apply training modifier to force and intelligence
    effective_force = side["force"] * (0.8 + 0.2 * side["training"])
    effective_intelligence = side["intelligence"] * (0.8 + 0.2 * side["training"])

    # Apply supply distance modifier to logistics
    effective_logistics = side["logistics"] * max(0.5, 1 - side["supply_distance"] / 1000)

    base_score = (
        0.25 * effective_force
        + 0.15 * side["mobility"]
        + 0.15 * effective_logistics
        + 0.15 * effective_intelligence
        + 0.10 * effective_communication
        + 0.20 * effective_readiness
    )

    score = base_score
    score += 0.06 * mobility_effective
    score += 0.06 * intel_effective
    score -= 0.1 * environmental_difficulty
    return score


def _terrain_mobility_weight(terrain: str) -> float:
    return {
        "Open": 1.0,
        "Hilly": 0.65,
        "Forest": 0.7,
        "Desert": 0.8,
        "Urban": 0.75,
        "Mixed": 0.85,
    }.get(terrain, 0.85)


def _apply_objective_modifiers(
    score_a: float,
    score_b: float,
    objective_type: str,
    terrain: str,
) -> tuple[float, float]:
    """Apply objective type modifiers to scores."""
    if objective_type == "Attack":
        score_a += 0.05
    elif objective_type == "Defend":
        if terrain in ["Hilly", "Forest", "Urban"]:
            score_b += 0.10
    elif objective_type == "Delay":
        # Reduces decisive thresholds - handled in outcome determination
        pass
    elif objective_type == "Withdraw":
        # Reduces stalemate probability - handled in outcome determination
        pass
    return score_a, score_b


def _resource_status(logistics: float, readiness: float, difficulty: float) -> str:
    resource_index = 0.55 * logistics + 0.45 * readiness - 0.25 * difficulty
    if resource_index >= 0.67:
        return "High"
    if resource_index >= 0.4:
        return "Medium"
    return "Low"


def _estimate_duration(
    differential: float,
    difficulty: float,
    terrain: str,
    weather: str,
    objective_type: str = "Attack",
) -> float:
    terrain_drag = {
        "Open": 0.2,
        "Hilly": 0.7,
        "Forest": 0.8,
        "Desert": 0.6,
        "Urban": 0.9,
        "Mixed": 0.5,
    }[terrain]
    weather_drag = {
        "Clear": 0.0,
        "Cloudy": 0.2,
        "Rain": 0.5,
        "Fog": 0.45,
        "Extreme": 0.75,
    }[weather]
    
    # Adjust balance drag based on objective type
    if objective_type == "Delay":
        balance_drag = max(0.0, 0.5 - abs(differential)) * 3.0  # More stalemates
    elif objective_type == "Withdraw":
        balance_drag = max(0.0, 0.25 - abs(differential)) * 3.0  # Less stalemates
    else:
        balance_drag = max(0.0, 0.35 - abs(differential)) * 3.0
    
    duration = 1.6 + terrain_drag + weather_drag + difficulty * 1.4 + balance_drag
    return round(float(max(1.0, duration)), 1)


def build_default_config() -> dict[str, Any]:
    """Default scenario used by tests and by the frontend later on."""
    return {
        "side_a_force_strength": 80,
        "side_b_force_strength": 70,
        "side_a_mobility": 0.75,
        "side_b_mobility": 0.60,
        "side_a_logistics": 0.70,
        "side_b_logistics": 0.55,
        "side_a_intelligence": 0.80,
        "side_b_intelligence": 0.60,
        "side_a_communication": 0.85,
        "side_b_communication": 0.65,
        "side_a_readiness": 0.78,
        "side_b_readiness": 0.62,
        "side_a_morale": 0.70,
        "side_b_morale": 0.60,
        "side_a_training": 0.75,
        "side_b_training": 0.65,
        "side_a_supply_distance": 200,
        "side_b_supply_distance": 250,
        "side_a_surprise": 0.30,
        "side_b_surprise": 0.25,
        "terrain_type": "Hilly",
        "weather": "Cloudy",
        "visibility": "Medium",
        "time_of_day": "Day",
        "environmental_difficulty": 0.35,
        "electronic_warfare_level": 0.25,
        "objective_type": "Attack",
    }

