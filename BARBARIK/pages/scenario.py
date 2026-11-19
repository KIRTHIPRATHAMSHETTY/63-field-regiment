"""
BARBARIK - New Scenario page
============================

Full-screen sliders for defining a fresh fictional scenario. On 'Run
Simulation' the scenario is simulated, predicted by the ML model and recorded
in the SQLite database. All outputs come from real function calls.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import ui_helpers as U
from database import save_scenario
from session import current_scenario_id, new_scenario
from session import ensure_initialized, reset_inputs
from simulation_engine import simulate_scenario


def render() -> None:
    ensure_initialized()
    st.title("New Scenario")
    st.caption("Create a fresh fictional scenario, then simulate + predict it.")

    s = st.session_state

    c_input, c_run = st.columns([1.5, 1])

    with c_input:
        left, right = st.columns(2)
        with left:
            st.markdown("#### SIDE A (FICTIONAL)")
            s["a_force"] = st.slider("Force Strength", 10, 100,
                                     int(s["a_force"]), step=1, key="new_a_force")
            s["a_mobility"] = st.slider("Mobility", 0.0, 1.0,
                                        float(s["a_mobility"]), key="new_a_mob")
            s["a_logistics"] = st.slider("Logistics", 0.0, 1.0,
                                         float(s["a_logistics"]), key="new_a_log")
            s["a_intelligence"] = st.slider("Intelligence", 0.0, 1.0,
                                            float(s["a_intelligence"]), key="new_a_int")
            s["a_communication"] = st.slider("Communication", 0.0, 1.0,
                                             float(s["a_communication"]), key="new_a_comm")
            s["a_readiness"] = st.slider("Readiness", 0.0, 1.0,
                                         float(s["a_readiness"]), key="new_a_read")
        with right:
            st.markdown("#### SIDE B (FICTIONAL)")
            s["b_force"] = st.slider("Force Strength", 10, 100,
                                     int(s["b_force"]), step=1, key="new_b_force")
            s["b_mobility"] = st.slider("Mobility", 0.0, 1.0,
                                        float(s["b_mobility"]), key="new_b_mob")
            s["b_logistics"] = st.slider("Logistics", 0.0, 1.0,
                                         float(s["b_logistics"]), key="new_b_log")
            s["b_intelligence"] = st.slider("Intelligence", 0.0, 1.0,
                                            float(s["b_intelligence"]), key="new_b_int")
            s["b_communication"] = st.slider("Communication", 0.0, 1.0,
                                             float(s["b_communication"]), key="new_b_comm")
            s["b_readiness"] = st.slider("Readiness", 0.0, 1.0,
                                         float(s["b_readiness"]), key="new_b_read")

        st.markdown("#### ENVIRONMENT")
        e1, e2 = st.columns(2)
        with e1:
            s["terrain"] = st.selectbox("Terrain", U.TERRAIN_OPTIONS,
                                        key="new_terrain")
            s["visibility"] = st.selectbox("Visibility", U.VISIBILITY_OPTIONS,
                                           key="new_visibility")
        with e2:
            s["weather"] = st.selectbox("Weather", U.WEATHER_OPTIONS,
                                        key="new_weather")
            s["time_of_day"] = st.selectbox("Time of Day", U.TIME_OPTIONS,
                                            key="new_time")
        s["env_difficulty"] = st.slider("Environmental Difficulty", 0.0, 1.0,
                                        float(s["env_difficulty"]), key="new_env")
        s["duration"] = st.slider("Simulation Duration (turns)", 1, 10,
                                  int(s["duration"]), key="new_dur")

    with c_run:
        st.markdown("#### ACTION")
        if st.button("RUN SIMULATION", type="primary", width="stretch"):
            _execute()
        if st.button("RESET", width="stretch"):
            reset_inputs()
            st.rerun()

        run = st.session_state.get("last_run")
        if run:
            st.markdown("---")
            st.markdown(f"**Last run: {run['scenario_id']}**")
            st.metric("Predicted Outcome", run["predicted"].replace("_", " "))
            probs = run["probs"]
            st.metric("P(Side A)", f"{probs.get('SIDE_A_ADVANTAGE', 0)*100:.1f}%")
            st.metric("P(Side B)", f"{probs.get('SIDE_B_ADVANTAGE', 0)*100:.1f}%")
            st.metric("P(Stalemate)", f"{probs.get('STALEMATE', 0)*100:.1f}%")


def _execute() -> None:
    """Simulate + predict the current input and store in the database."""
    from model import predict_outcome

    model, pipeline = U.get_model_and_pipeline()
    if model is None:
        st.error("MODEL NOT TRAINED — Please generate dataset and train model.")
        return

    s = st.session_state
    config = U.build_config_from_state(s)
    sim = simulate_scenario(config)  # REAL simulation

    raw_series = U.build_input_series(s)
    X = pipeline.transform(raw_series.to_frame().T)
    probs = model.predict_proba(X)[0]
    prob_dict = {c: float(p) for c, p in zip(model.classes_, probs)}
    predicted = model.classes_[int(probs.argmax())]

    sid = current_scenario_id()
    save_scenario(
        scenario_id=sid,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        side_a={
            "force_strength": int(s["a_force"]),
            "mobility": float(s["a_mobility"]),
            "logistics": float(s["a_logistics"]),
            "intelligence": float(s["a_intelligence"]),
            "communication": float(s["a_communication"]),
            "readiness": float(s["a_readiness"]),
        },
        side_b={
            "force_strength": int(s["b_force"]),
            "mobility": float(s["b_mobility"]),
            "logistics": float(s["b_logistics"]),
            "intelligence": float(s["b_intelligence"]),
            "communication": float(s["b_communication"]),
            "readiness": float(s["b_readiness"]),
        },
        environment={
            "terrain": s["terrain"],
            "weather": s["weather"],
            "visibility": s["visibility"],
            "time_of_day": s["time_of_day"],
            "environmental_difficulty": float(s["env_difficulty"]),
            "simulation_duration": int(s["duration"]),
        },
        predicted_outcome=predicted,
        prob_a=prob_dict.get("SIDE_A_ADVANTAGE", 0.0),
        prob_b=prob_dict.get("SIDE_B_ADVANTAGE", 0.0),
        prob_draw=prob_dict.get("STALEMATE", 0.0),
    )

    st.session_state["last_run"] = {
        "scenario_id": sid,
        "sim": sim,
        "probs": prob_dict,
        "predicted": predicted,
        "top_features": [],
    }
    new_scenario()