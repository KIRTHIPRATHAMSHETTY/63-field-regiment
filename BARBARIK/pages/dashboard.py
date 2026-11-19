"""
BARBARIK - Dashboard page
=========================

The main dashboard: Scenario Input | Simulated Battlefield | Predicted Outcome,
plus a bottom row with SHAP importance, live model metrics and scenario history.

Layout (as specified):
    st.columns([1, 1.4, 1])   -> input | battlefield | outcome
    st.columns([1, 1, 1])     -> SHAP | model metrics | history
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import ui_helpers as U
from session import ensure_initialized, current_scenario_id, reset_inputs, new_scenario
from simulation_engine import simulate_scenario
from database import save_scenario, get_all_scenarios


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------
def _run_scenario() -> None:
    """Execute a full run: simulate, predict, explain, record. Real calls only."""
    model, pipeline = U.get_model_and_pipeline()
    if model is None:
        st.session_state["run_error"] = "MODEL NOT TRAINED"
        return

    config = U.build_config_from_state(st.session_state)
    sim = simulate_scenario(config)  # REAL simulation

    # Preprocess the current input and predict (REAL model call).
    raw_series = U.build_input_series(st.session_state)
    X = pipeline.transform(raw_series.to_frame().T)
    probs = model.predict_proba(X)[0]
    prob_dict = {c: float(p) for c, p in zip(model.classes_, probs)}
    predicted = model.classes_[int(probs.argmax())]

    # Real local SHAP for the predicted class (cached explainer; fast).
    try:
        expl = U.get_shap_explainer_cached()
        if expl is not None:
            class_idx = int(list(model.classes_).index(predicted))
            sv = U._local_shap_for_X(expl, X, class_index=class_idx)
            feature_names = U.get_feature_names()
            top_features = _top8(sv, feature_names)
        else:
            top_features = []
    except Exception:
        top_features = []

    sid = current_scenario_id()

    # Persist to SQLite history (src/database.py).
    save_scenario(
        scenario_id=sid,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        side_a=_side_dict(st.session_state, "a"),
        side_b=_side_dict(st.session_state, "b"),
        environment=_env_dict(st.session_state),
        predicted_outcome=predicted,
        prob_a=prob_dict.get("SIDE_A_ADVANTAGE", 0.0),
        prob_b=prob_dict.get("SIDE_B_ADVANTAGE", 0.0),
        prob_draw=prob_dict.get("STALEMATE", 0.0),
    )

    st.session_state["run_error"] = None
    st.session_state["last_run"] = {
        "scenario_id": sid,
        "sim": sim,
        "probs": prob_dict,
        "predicted": predicted,
        "top_features": top_features,
    }

    # Advance the scenario counter for the next NEW scenario.
    new_scenario()


# ---------------------------------------------------------------------------
# Left column
# ---------------------------------------------------------------------------
def _render_input(col) -> None:
    with col:
        st.markdown('<div class="card"><h3>Scenario Input</h3></div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### SIDE A (FICTIONAL)")
            st.session_state["a_force"] = st.slider(
                "A · Force Strength", 10, 100, int(st.session_state["a_force"]),
                step=1, key="s_a_force")
            st.session_state["a_mobility"] = st.slider(
                "A · Mobility", 0.0, 1.0, float(st.session_state["a_mobility"]),
                key="s_a_mobility")
            st.session_state["a_logistics"] = st.slider(
                "A · Logistics", 0.0, 1.0, float(st.session_state["a_logistics"]),
                key="s_a_logistics")
            st.session_state["a_intelligence"] = st.slider(
                "A · Intelligence", 0.0, 1.0,
                float(st.session_state["a_intelligence"]), key="s_a_intelligence")
            st.session_state["a_communication"] = st.slider(
                "A · Communication", 0.0, 1.0,
                float(st.session_state["a_communication"]), key="s_a_communication")
            st.session_state["a_readiness"] = st.slider(
                "A · Readiness", 0.0, 1.0,
                float(st.session_state["a_readiness"]), key="s_a_readiness")
            st.session_state["a_morale"] = st.slider(
                "A · Morale", 0.0, 1.0,
                float(st.session_state.get("a_morale", 0.7)), key="s_a_morale")
            st.session_state["a_training"] = st.slider(
                "A · Training", 0.0, 1.0,
                float(st.session_state.get("a_training", 0.75)), key="s_a_training")
            st.session_state["a_supply_distance"] = st.slider(
                "A · Supply Distance (km)", 0, 500,
                int(st.session_state.get("a_supply_distance", 200)), key="s_a_supply_distance")
            st.session_state["a_surprise"] = st.slider(
                "A · Surprise", 0.0, 1.0,
                float(st.session_state.get("a_surprise", 0.3)), key="s_a_surprise")

        with st.container(border=True):
            st.markdown("#### SIDE B (FICTIONAL)")
            st.session_state["b_force"] = st.slider(
                "B · Force Strength", 10, 100, int(st.session_state["b_force"]),
                step=1, key="s_b_force")
            st.session_state["b_mobility"] = st.slider(
                "B · Mobility", 0.0, 1.0, float(st.session_state["b_mobility"]),
                key="s_b_mobility")
            st.session_state["b_logistics"] = st.slider(
                "B · Logistics", 0.0, 1.0, float(st.session_state["b_logistics"]),
                key="s_b_logistics")
            st.session_state["b_intelligence"] = st.slider(
                "B · Intelligence", 0.0, 1.0,
                float(st.session_state["b_intelligence"]), key="s_b_intelligence")
            st.session_state["b_communication"] = st.slider(
                "B · Communication", 0.0, 1.0,
                float(st.session_state["b_communication"]), key="s_b_communication")
            st.session_state["b_readiness"] = st.slider(
                "B · Readiness", 0.0, 1.0,
                float(st.session_state["b_readiness"]), key="s_b_readiness")
            st.session_state["b_morale"] = st.slider(
                "B · Morale", 0.0, 1.0,
                float(st.session_state.get("b_morale", 0.6)), key="s_b_morale")
            st.session_state["b_training"] = st.slider(
                "B · Training", 0.0, 1.0,
                float(st.session_state.get("b_training", 0.65)), key="s_b_training")
            st.session_state["b_supply_distance"] = st.slider(
                "B · Supply Distance (km)", 0, 500,
                int(st.session_state.get("b_supply_distance", 250)), key="s_b_supply_distance")
            st.session_state["b_surprise"] = st.slider(
                "B · Surprise", 0.0, 1.0,
                float(st.session_state.get("b_surprise", 0.25)), key="s_b_surprise")

        with st.container(border=True):
            st.markdown("#### ENVIRONMENT")
            st.session_state["terrain"] = st.selectbox(
                "Terrain", U.TERRAIN_OPTIONS,
                index=U.TERRAIN_OPTIONS.index(st.session_state["terrain"]),
                key="s_terrain")
            st.session_state["weather"] = st.selectbox(
                "Weather", U.WEATHER_OPTIONS,
                index=U.WEATHER_OPTIONS.index(st.session_state["weather"]),
                key="s_weather")
            st.session_state["visibility"] = st.selectbox(
                "Visibility", U.VISIBILITY_OPTIONS,
                index=U.VISIBILITY_OPTIONS.index(st.session_state["visibility"]),
                key="s_visibility")
            st.session_state["time_of_day"] = st.selectbox(
                "Time of Day", U.TIME_OPTIONS,
                index=U.TIME_OPTIONS.index(st.session_state["time_of_day"]),
                key="s_time")
            st.session_state["env_difficulty"] = st.slider(
                "Environmental Difficulty", 0.0, 1.0,
                float(st.session_state["env_difficulty"]), key="s_env")
            st.session_state["duration"] = st.slider(
                "Simulation Duration (turns)", 1, 10,
                int(st.session_state["duration"]), key="s_dur")
            ew_mapping = {"None": 0.0, "Low": 0.25, "Moderate": 0.5, "High": 0.75, "Severe": 1.0}
            ew_label = next((k for k, v in ew_mapping.items() if v == st.session_state.get("ew_level", 0.25)), "Low")
            ew_selected = st.selectbox(
                "Electronic Warfare Level", U.EW_OPTIONS,
                index=U.EW_OPTIONS.index(ew_label),
                key="s_ew")
            st.session_state["ew_level"] = ew_mapping[ew_selected]

        with st.container(border=True):
            st.markdown("#### MISSION")
            st.session_state["objective_type"] = st.selectbox(
                "Objective Type", U.OBJECTIVE_OPTIONS,
                index=U.OBJECTIVE_OPTIONS.index(st.session_state.get("objective_type", "Attack")),
                key="s_objective")

        # ---- Buttons --------------------------------------------------------
        c_run, c_res, c_load = st.columns(3)
        with c_run:
            run_pressed = st.button("RUN SIMULATION", type="primary", width="stretch")
        with c_res:
            if st.button("RESET", width="stretch"):
                reset_inputs()
                st.rerun()
        with c_load:
            st.button("LOAD SCENARIO", width="stretch", disabled=True)

        if run_pressed:
            _run_scenario()

        if st.session_state.get("run_error"):
            st.error("MODEL NOT TRAINED — Please generate dataset and train model.")


# ---------------------------------------------------------------------------
# Center column
# ---------------------------------------------------------------------------
def _render_battlefield(col) -> None:
    with col:
        run = st.session_state.get("last_run")
        sid = (run["scenario_id"] if run else current_scenario_id())
        fig = U.make_battlefield_figure(sid, st.session_state)
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Right column
# ---------------------------------------------------------------------------
def _render_outcome(col) -> None:
    with col:
        st.markdown('<div class="card"><h3>Predicted Outcome</h3></div>',
                    unsafe_allow_html=True)
        run = st.session_state.get("last_run")

        if run is None:
            st.info("Run a scenario to see the predicted outcome.")
            return

        predicted = run["predicted"]
        cls = U.outcome_badge_class(predicted)
        label = {
            "SIDE_A_ADVANTAGE": "SIDE A ADVANTAGE",
            "SIDE_B_ADVANTAGE": "SIDE B ADVANTAGE",
            "STALEMATE": "STALEMATE",
        }[predicted]

        st.markdown(
            f'<div class="outcome-big {cls}">{label}</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(U.make_donut_figure(run["probs"]), width="stretch")

        st.markdown(
            '<div class="card"><h3>Simulated Metrics</h3>'
            '<div class="subtitle">from simulate_scenario()</div></div>',
            unsafe_allow_html=True,
        )
        sim = run["sim"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Duration", f"{sim['duration']} turns")
        m2.metric("Resource A", sim["resource_status_a"])
        m3.metric("Resource B", sim["resource_status_b"])


# ---------------------------------------------------------------------------
# Bottom row
# ---------------------------------------------------------------------------
def _render_shap(col) -> None:
    with col:
        st.markdown('<div class="card"><h3>SHAP Feature Importance</h3></div>',
                    unsafe_allow_html=True)
        run = st.session_state.get("last_run")
        features = (run["top_features"] if run else [])

        if not features:
            st.info("Run a scenario to see SHAP attributions.")
            if st.button("VIEW DETAILED XAI ANALYSIS", disabled=True):
                pass
            return

        names = [f[0] for f in features][::-1]
        values = [f[1] for f in features][::-1]
        colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in values]

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h", marker_color=colors,
        ))
        fig.update_layout(
            height=320,
            xaxis_title="SHAP value",
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e9f0"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.button("VIEW DETAILED XAI ANALYSIS")


def _render_model_metrics(col) -> None:
    with col:
        st.markdown('<div class="card"><h3>Model Performance</h3>'
                    '<div class="subtitle">Random Forest Classifier</div></div>',
                    unsafe_allow_html=True)
        model, _ = U.get_model_and_pipeline()
        if model is None:
            st.error("MODEL NOT TRAINED — Please generate dataset and train model.")
            if st.button("TRAIN MODEL"):
                st.code("Run:  python train_model.py", language="bash")
            return

        m = U.get_model_metrics()  # REAL evaluate_model() output, cached
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.3f}")
        b.metric("Precision", f"{m['precision_macro']:.3f}")
        c.metric("Recall", f"{m['recall_macro']:.3f}")
        d.metric("F1", f"{m['f1_macro']:.3f}")
        e.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] else "n/a")


def _render_history(col) -> None:
    with col:
        st.markdown('<div class="card"><h3>Scenario History</h3></div>',
                    unsafe_allow_html=True)
        rows = get_all_scenarios()[:5]
        if rows:
            st.dataframe(
                [
                    {
                        "ID": r["id"], "Timestamp": r["timestamp"],
                        "Outcome": r["predicted_outcome"],
                        "A%": round(r["prob_a"] * 100, 1),
                        "B%": round(r["prob_b"] * 100, 1),
                        "Draw%": round(r["prob_draw"] * 100, 1),
                    }
                    for r in rows
                ],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("No scenarios recorded yet.")
        st.button("VIEW ALL")


def _top8(shap_vector, feature_names):
    """Top-8 features by absolute SHAP value: [(name, sign_val), ...]."""
    from explainability import get_top_influencing_features
    if shap_vector is None:
        return []
    return get_top_influencing_features(shap_vector, feature_names, top_n=8)


def _side_dict(state, side: str) -> dict:
    """Extract one side's attributes from session state -> DB JSON."""
    p = "a" if side == "a" else "b"
    return {
        "force_strength": int(state[f"{p}_force"]),
        "mobility": float(state[f"{p}_mobility"]),
        "logistics": float(state[f"{p}_logistics"]),
        "intelligence": float(state[f"{p}_intelligence"]),
        "communication": float(state[f"{p}_communication"]),
        "readiness": float(state[f"{p}_readiness"]),
        "morale": float(state.get(f"{p}_morale", 0.7 if p == "a" else 0.6)),
        "training": float(state.get(f"{p}_training", 0.75 if p == "a" else 0.65)),
        "supply_distance": float(state.get(f"{p}_supply_distance", 200 if p == "a" else 250)),
        "surprise": float(state.get(f"{p}_surprise", 0.3 if p == "a" else 0.25)),
    }


def _env_dict(state) -> dict:
    """Extract environment attributes from session state -> DB JSON."""
    return {
        "terrain": state["terrain"],
        "weather": state["weather"],
        "visibility": state["visibility"],
        "time_of_day": state["time_of_day"],
        "environmental_difficulty": float(state["env_difficulty"]),
        "simulation_duration": int(state["duration"]),
        "electronic_warfare_level": float(state.get("ew_level", 0.25)),
        "objective_type": state.get("objective_type", "Attack"),
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render() -> None:
    ensure_initialized()
    st.title("Dashboard")

    col_left, col_center, col_right = st.columns([1, 1.4, 1])
    _render_input(col_left)
    _render_battlefield(col_center)
    _render_outcome(col_right)

    st.markdown("---")
    bl, bc, br = st.columns([1, 1, 1])
    _render_shap(bl)
    _render_model_metrics(bc)
    _render_history(br)