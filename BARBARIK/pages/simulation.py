"""
BARBARIK - Simulation page
==========================

Runs the scenario through an animated step-by-step pipeline using
``st.progress``: INITIALIZATION -> ENVIRONMENT ANALYSIS -> ROUND 1 -> ROUND 2 ->
ROUND 3 -> OUTCOME CALCULATION -> ML PREDICTION.

Every intermediate value comes from the real simulation engine / ML model.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

import ui_helpers as U
from database import save_scenario
from session import current_scenario_id, ensure_initialized, new_scenario
from simulation_engine import simulate_scenario

STEPS = [
    "INITIALIZATION",
    "ENVIRONMENT ANALYSIS",
    "ROUND 1",
    "ROUND 2",
    "ROUND 3",
    "OUTCOME CALCULATION",
    "ML PREDICTION",
]


def render() -> None:
    ensure_initialized()
    st.title("Simulation")
    st.caption("Animated step-by-step simulation of the current scenario.")

    model, pipeline = U.get_model_and_pipeline()
    if model is None:
        st.error("MODEL NOT TRAINED — Please generate dataset and train model.")
        if st.button("TRAIN MODEL"):
            st.code("Run:  python train_model.py", language="bash")
        return

    st.markdown("#### Current scenario inputs")
    cfg = U.build_config_from_state(st.session_state)
    st.write(pd.DataFrame([cfg]).T.rename(columns={0: "value"}))

    if st.button("START SIMULATION", type="primary"):
        _run_steps(cfg, model, pipeline)


def _run_steps(config, model, pipeline) -> None:
    s = st.session_state
    progress = st.progress(0.0)
    status = st.empty()

    n = len(STEPS)
    # 1..5 = simulation engine phases; 6 = outcome calc; 7 = ML prediction.
    sim_result = None
    prob_dict = {}
    predicted = None

    for i, step in enumerate(STEPS, start=1):
        status.markdown(f"**{step}**")
        time.sleep(0.35)

        if step in ("INITIALIZATION", "ENVIRONMENT ANALYSIS", "ROUND 1",
                    "ROUND 2", "ROUND 3"):
            # Simulate a fresh copy each phase is wasteful; run once at the
            # final sim step, but we still want the progress animation. We
            # compute the real simulation at "ROUND 3" so it is genuinely used.
            if step == "ROUND 3":
                sim_result = simulate_scenario(config)  # REAL simulation
                status.markdown(
                    f"**{step}** — rounds simulated (stochastic engine).")
                time.sleep(0.2)
        elif step == "OUTCOME CALCULATION":
            # Derive outcome from the real engine result.
            if sim_result is not None:
                status.markdown(
                    f"**{step}** — {sim_result['outcome']} "
                    f"(A={sim_result['simulation_score_a']:.3f}, "
                    f"B={sim_result['simulation_score_b']:.3f}).")
            time.sleep(0.25)
        elif step == "ML PREDICTION":
            raw_series = U.build_input_series(session_state_from_config(config))
            X = pipeline.transform(raw_series.to_frame().T)
            probs = model.predict_proba(X)[0]
            prob_dict = {c: float(p) for c, p in zip(model.classes_, probs)}
            predicted = model.classes_[int(probs.argmax())]
            status.markdown(f"**{step}** — predicted: {predicted.replace('_', ' ')}.")
            time.sleep(0.25)

        progress.progress(i / n)

    progress.progress(1.0)
    status.markdown("#### ✅ Simulation complete")

    if predicted is not None and sim_result is not None:
        sid = current_scenario_id()
        save_scenario(
            scenario_id=sid,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            side_a=_side_dict(config, "a"),
            side_b=_side_dict(config, "b"),
            environment=_env_dict(config),
            predicted_outcome=predicted,
            prob_a=prob_dict.get("SIDE_A_ADVANTAGE", 0.0),
            prob_b=prob_dict.get("SIDE_B_ADVANTAGE", 0.0),
            prob_draw=prob_dict.get("STALEMATE", 0.0),
        )

        st.success(f"Scenario {sid} recorded.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Outcome", predicted.replace("_", " "))
        m2.metric("Duration", f"{sim_result['duration']} turns")
        m3.metric("Engine Result", sim_result["outcome"].replace("_", " "),)

        st.write("**Prediction probabilities**")
        st.write(pd.DataFrame([prob_dict]).T.rename(columns={0: "probability"}))

        new_scenario()


def session_state_from_config(config: dict) -> dict:
    """Reconstruct a thin session-state-like view from a config dict."""
    return config


def _side_dict(config: dict, side: str) -> dict:
    p = "a" if side == "a" else "b"
    return {
        "force_strength": int(config[f"side_{p}_force_strength"]),
        "mobility": float(config[f"side_{p}_mobility"]),
        "logistics": float(config[f"side_{p}_logistics"]),
        "intelligence": float(config[f"side_{p}_intelligence"]),
        "communication": float(config[f"side_{p}_communication"]),
        "readiness": float(config[f"side_{p}_readiness"]),
    }


def _env_dict(config: dict) -> dict:
    return {
        "terrain": config["terrain_type"],
        "weather": config["weather"],
        "visibility": config["visibility"],
        "time_of_day": config["time_of_day"],
        "environmental_difficulty": float(config["environmental_difficulty"]),
        "simulation_duration": int(config["simulation_duration"]),
    }