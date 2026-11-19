"""
BARBARIK - About page
=====================
"""

from __future__ import annotations
import streamlit as st


def render() -> None:
    st.title("About BARBARIK")

    st.markdown(
        """
## Project Description

**BARBARIK** is an academic AIML mini-project that couples a *fictional*
wargaming simulation engine with a machine-learning outcome predictor. It
takes abstract scenario attributes for two sides, runs a stochastic,
rule-based simulation, and trains a Random Forest to predict the resulting
outcome class from the raw inputs.

## Ethical Scope

- **Fiction only.** All scenarios, forces, terrains and outcomes are invented
  and carry **no** relation to any real military data, targets, borders, or
  real-world conflict.
- **Educational intent.** The purpose is to demonstrate the full ML lifecycle
  — data generation, preprocessing, training, evaluation, explainability and
  deployment — not to model real warfare.
- **No policy use.** Nothing in this project should be used to make decisions
  about real conflicts or operations.

## Tech Stack

- **Streamlit** — interactive dashboard
- **scikit-learn** — Random Forest, preprocessing, metrics
- **pandas / numpy** — data handling
- **SHAP** — model explainability
- **matplotlib / plotly** — visualisations
- **seaborn / joblib** — styling helpers and model persistence
- **SQLite (stdlib)** — scenario history store

## Limitations

- The simulation scoring weights are hand-designed; outcomes reflect those
  choices rather than real doctrine.
- Stochastic noise (σ ≈ 0.05) bounds prediction accuracy at roughly ~0.79 —
  irreducible uncertainty by design.
- SHAP computing on a 300-tree forest is relatively slow in the dashboard.
- The dataset is synthetic, so performance does not transfer to real data.
- Single-machine demo; not production-hardened.
"""
    )