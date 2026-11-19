"""
BARBARIK - Session state helpers
================================

Initialises the dashboard's scenario-input session state and manages the
current scenario ID counter.
"""

from __future__ import annotations

import streamlit as st

DEFAULTS = {
    "a_force": 50.0,
    "b_force": 50.0,
    "a_mobility": 0.5,
    "b_mobility": 0.5,
    "a_logistics": 0.5,
    "b_logistics": 0.5,
    "a_intelligence": 0.5,
    "b_intelligence": 0.5,
    "a_communication": 0.5,
    "b_communication": 0.5,
    "a_readiness": 0.5,
    "b_readiness": 0.5,
    "a_morale": 0.7,
    "b_morale": 0.6,
    "a_training": 0.75,
    "b_training": 0.65,
    "a_supply_distance": 200.0,
    "b_supply_distance": 250.0,
    "a_surprise": 0.3,
    "b_surprise": 0.25,
    "terrain": "Open",
    "weather": "Clear",
    "visibility": "High",
    "time_of_day": "Day",
    "env_difficulty": 0.3,
    "duration": 5,
    "ew_level": 0.25,
    "objective_type": "Attack",
}


def ensure_initialized() -> None:
    """Populate session state with defaults if not already present."""
    if "scenario_counter" not in st.session_state:
        st.session_state["scenario_counter"] = 1

    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_scenario_id() -> str:
    """Return the current simulation scenario id (e.g. 'SCN-0001')."""
    ensure_initialized()
    return f"SCN-{st.session_state['scenario_counter']:04d}"


def new_scenario() -> None:
    """Generate the next scenario id."""
    ensure_initialized()
    st.session_state["scenario_counter"] += 1


def reset_inputs() -> None:
    """Reset all scenario inputs to balanced defaults."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value