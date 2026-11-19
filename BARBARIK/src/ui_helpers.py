"""
BARBARIK - shared cached resources for the Streamlit dashboard
==============================================================

Centralises the caching policy required by the spec:
  * model + pipeline  -> @st.cache_resource (heavy, non-serialisable objects)
  * dataset / metrics -> @st.cache_data (DataFrames, plain values)

Every value exposed here comes from a real function call — nothing hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import load_model
from preprocessing import (
    load_dataset,
    split_data,
    build_preprocessing_pipeline,
)

# --------------------------- page constants -------------------------------
PAGE_NAMES = [
    "Dashboard",
    "New Scenario",
    "Simulation",
    "Dataset",
    "Model Performance",
    "Explainable AI",
    "Scenario History",
    "About",
]

SHAP_BACKGROUND_SIZE = 300

# Configuration ranges / options mirrored from the simulation engine.
TERRAIN_OPTIONS = ["Open", "Hilly", "Forest", "Desert", "Urban", "Mixed"]
WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Extreme"]
VISIBILITY_OPTIONS = ["High", "Medium", "Low"]
TIME_OPTIONS = ["Day", "Night"]
OBJECTIVE_OPTIONS = ["Attack", "Defend", "Delay", "Withdraw"]
EW_OPTIONS = ["None", "Low", "Moderate", "High", "Severe"]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model + pipeline...")
def get_model_and_pipeline():
    """Load the persisted RandomForest model and preprocessing pipeline."""
    if not __import__("os").path.exists("models/random_forest_model.joblib"):
        return None, None
    return load_model("models/")


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset() -> pd.DataFrame:
    return load_dataset("data/barbarik_dataset.csv")


@st.cache_data(show_spinner="Evaluating model on test data...")
def get_dataset_splits():
    """Return (X_train, X_test, y_train, y_test) split from the cached data."""
    df = get_dataset()
    return split_data(df)


@st.cache_data(show_spinner="Computing model metrics...")
def get_model_metrics():
    """Real sklearn evaluation of the model on the held-out test split."""
    from model import evaluate_model

    model, pipeline = get_model_and_pipeline()
    X_train, X_test, y_train, y_test = get_dataset_splits()

    X_test_pp = pipeline.transform(X_test)
    return evaluate_model(model, X_test_pp, y_test)


@st.cache_data(show_spinner="Preprocessing background data for SHAP...")
def get_transformed_splits():
    """Preprocessed train/test used for SHAP explanations."""
    model, pipeline = get_model_and_pipeline()
    X_train, X_test, y_train, y_test = get_dataset_splits()
    X_train_pp = pipeline.transform(X_train)
    X_test_pp = pipeline.transform(X_test)
    return X_train_pp, X_test_pp, y_test


def get_feature_names() -> list[str]:
    """The preprocessed (post-pipeline) feature names, in column order."""
    _, pipeline = get_model_and_pipeline()
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------
def build_config_from_state(state) -> dict:
    """Assemble a simulation-engine config dict from Streamlit session state."""
    return {
        "side_a_force_strength": int(state["a_force"]),
        "side_b_force_strength": int(state["b_force"]),
        "side_a_mobility": float(state["a_mobility"]),
        "side_b_mobility": float(state["b_mobility"]),
        "side_a_logistics": float(state["a_logistics"]),
        "side_b_logistics": float(state["b_logistics"]),
        "side_a_intelligence": float(state["a_intelligence"]),
        "side_b_intelligence": float(state["b_intelligence"]),
        "side_a_communication": float(state["a_communication"]),
        "side_b_communication": float(state["b_communication"]),
        "side_a_readiness": float(state["a_readiness"]),
        "side_b_readiness": float(state["b_readiness"]),
        "side_a_morale": float(state.get("a_morale", 0.7)),
        "side_b_morale": float(state.get("b_morale", 0.6)),
        "side_a_training": float(state.get("a_training", 0.75)),
        "side_b_training": float(state.get("b_training", 0.65)),
        "side_a_supply_distance": float(state.get("a_supply_distance", 200)),
        "side_b_supply_distance": float(state.get("b_supply_distance", 250)),
        "side_a_surprise": float(state.get("a_surprise", 0.3)),
        "side_b_surprise": float(state.get("b_surprise", 0.25)),
        "terrain_type": state["terrain"],
        "weather": state["weather"],
        "visibility": state["visibility"],
        "time_of_day": state["time_of_day"],
        "environmental_difficulty": float(state.get("env_difficulty", 0.3)),
        "electronic_warfare_level": float(state.get("ew_level", 0.25)),
        "objective_type": state.get("objective_type", "Attack"),
    }


def build_input_series(state) -> pd.Series:
    """A single raw input row (features only) for the preprocessing pipeline."""
    config = build_config_from_state(state)
    keys = [
        "side_a_force_strength", "side_b_force_strength",
        "side_a_mobility", "side_b_mobility",
        "side_a_logistics", "side_b_logistics",
        "side_a_intelligence", "side_b_intelligence",
        "side_a_communication", "side_b_communication",
        "side_a_readiness", "side_b_readiness",
        "side_a_morale", "side_b_morale",
        "side_a_training", "side_b_training",
        "side_a_supply_distance", "side_b_supply_distance",
        "side_a_surprise", "side_b_surprise",
        "terrain_type", "weather", "visibility", "time_of_day",
        "environmental_difficulty", "electronic_warfare_level", "objective_type",
    ]
    return pd.Series({k: config[k] for k in keys})


def make_battlefield_figure(
    scenario_id: str,
    state,
) -> go.Figure:
    """Build the abstract 'Simulated Battlefield' Plotly scatter.

    Purely abstract shapes: colored zone bands, terrain-tinted background,
    Side A markers on the left, Side B markers on the right. No real map data.
    """
    terrain = state.get("terrain", "Mixed")
    _terrain_tints = {
        "Open": (0.18, 0.22, 0.32),
        "Hilly": (0.16, 0.20, 0.30),
        "Forest": (0.13, 0.20, 0.26),
        "Desert": (0.22, 0.20, 0.24),
        "Urban": (0.17, 0.18, 0.28),
        "Mixed": (0.16, 0.20, 0.30),
    }
    tint = _terrain_tints.get(terrain, _terrain_tints["Mixed"])

    rng = np.random.default_rng(abs(hash(scenario_id)) % (10**8))

    n_a = 10
    n_b = 10
    a_x = rng.uniform(0.5, 2.2, n_a)
    a_y = rng.uniform(0.5, 9.5, n_a)
    b_x = rng.uniform(7.8, 9.5, n_b)
    b_y = rng.uniform(0.5, 9.5, n_b)

    fig = go.Figure()

    # Abstract terrain-colored background bands for the three zones.
    fig.add_hrect(y0=0, y1=3.3, fillcolor="rgba(255,255,255,0.02)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=3.3, y1=6.6, fillcolor="rgba(255,255,255,0.03)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=6.6, y1=10, fillcolor="rgba(255,255,255,0.02)",
                  line_width=0, layer="below")

    # Zone separator / labels across the middle.
    for zx, label in [(3.5, "ZONE 1"), (7.0, "ZONE 2"), (10.5, "ZONE 3")]:
        pass  # labels drawn via annotations below

    # Side A (left, blue) and Side B (right, red) markers.
    fig.add_trace(go.Scatter(
        x=a_x, y=a_y, mode="markers", name="SIDE A",
        marker=dict(size=11, color="#3b82f6", opacity=0.85,
                    line=dict(width=1, color="#3b82f6")),
    ))
    fig.add_trace(go.Scatter(
        x=b_x, y=b_y, mode="markers", name="SIDE B",
        marker=dict(size=11, color="#ef4444", opacity=0.85,
                    line=dict(width=1, color="#ef4444")),
    ))

    # Abstract mid-field zone lines.
    fig.add_vline(x=5, line_width=1, line_dash="dash",
                  line_color="rgba(139,149,167,0.5)")

    annotations = [
        dict(x=1.2, y=9.7, text="SIDE A", showarrow=False,
             font=dict(color="#3b82f6", size=11, weight="bold")),
        dict(x=9.0, y=9.7, text="SIDE B", showarrow=False,
             font=dict(color="#ef4444", size=11, weight="bold")),
        dict(x=5.0, y=9.75, text="ZONE 1 — ZONE 2 — ZONE 3", showarrow=False,
             font=dict(color="#8b95a7", size=10)),
    ]
    for a in annotations:
        fig.add_annotation(**a)

    fig.update_layout(
        title=f"SIMULATED BATTLEFIELD — SCENARIO {scenario_id}",
        xaxis=dict(visible=False, range=[0, 10]),
        yaxis=dict(visible=False, range=[0, 10]),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=f"rgb({int(tint[0]*255)},{int(tint[1]*255)},{int(tint[2]*255)})",
        font=dict(color="#e5e9f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def outcome_badge_class(outcome: str) -> str:
    """Return the CSS class used to colour the big outcome label."""
    if outcome == "SIDE_A_ADVANTAGE":
        return "outcome-a outcome-panel-a"
    if outcome == "SIDE_B_ADVANTAGE":
        return "outcome-b outcome-panel-b"
    return "outcome-s outcome-panel-s"


def make_donut_figure(probs: dict) -> go.Figure:
    """Donut chart of the model's predict_proba() output (3 slices)."""
    labels = ["SIDE A ADV", "SIDE B ADV", "STALEMATE"]
    values = [
        probs.get("SIDE_A_ADVANTAGE", 0.0),
        probs.get("SIDE_B_ADVANTAGE", 0.0),
        probs.get("STALEMATE", 0.0),
    ]
    colors = ["#3b82f6", "#ef4444", "#10b981"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55, sort=False,
        marker=dict(colors=colors),
        textinfo="percent", textfont=dict(color="#e5e9f0", size=12),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="MODEL PREDICTION PROBABILITY",
        height=300,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e9f0"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached SHAP resources (real computations only; never fabricated).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Computing SHAP explainer...")
def get_shap_explainer_cached():
    """A cached SHAP TreeExplainer for the trained model."""
    from explainability import get_shap_explainer

    model, _ = get_model_and_pipeline()
    if model is None:
        return None
    X_train_pp, _ = get_transformed_splits()
    return get_shap_explainer(model, X_train_pp, n_background=SHAP_BACKGROUND_SIZE)


@st.cache_data(show_spinner="Computing global SHAP values...")
def get_global_shap_cached(sample_size: int = 60):
    """Per-class SHAP values for a sample of test rows (list over classes).

    ``sample_size`` is kept modest because SHAP on a 300-tree Random Forest is
    comparatively expensive; 60 rows gives a clean beeswarm summary without
    hanging the dashboard.
    """
    from explainability import get_global_shap_values

    explainer = get_shap_explainer_cached()
    _, X_test_pp, _ = get_transformed_splits()
    X_sample = X_test_pp[:sample_size]
    return get_global_shap_values(explainer, X_sample)


def get_local_shap_for_scenario(state):
    """Local SHAP vector for the *current* scenario's predicted class.

    Returns (predicted, shap_vector, top_features, scaled_row) where
    ``scaled_row`` is the post-pipeline feature row used for SHAP waterfall.
    """
    from explainability import get_local_shap_values

    model, pipeline = get_model_and_pipeline()
    explainer = get_shap_explainer_cached()
    if model is None or explainer is None:
        return None, None, [], None

    raw_series = build_input_series(state)
    X = pipeline.transform(raw_series.to_frame().T)
    probs = model.predict_proba(X)[0]
    predicted = model.classes_[int(probs.argmax())]
    class_idx = int(list(model.classes_).index(predicted))

    sv = get_local_shap_values(explainer, X, class_index=class_idx)
    feature_names = get_feature_names()

    from explainability import get_top_influencing_features
    top = get_top_influencing_features(sv, feature_names, top_n=8)
    return predicted, sv, top, X


def _local_shap_for_X(explainer, X, class_index: int = 0):
    """Local SHAP vector (1-D) for a preprocessed single row and class."""
    from explainability import get_local_shap_values
    return get_local_shap_values(explainer, X, class_index=class_index)