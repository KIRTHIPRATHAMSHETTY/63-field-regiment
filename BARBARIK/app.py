"""
BARBARIK — Streamlit dashboard (main entry point)
=================================================

Fictional wargaming simulation + ML prediction dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from session import ensure_initialized, current_scenario_id  # noqa: E402

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="BARBARIK",
    page_icon="🛡",
    initial_sidebar_state="expanded",
)

# Shared session state for scenario inputs.
ensure_initialized()

# Global CSS theme.
_css = (Path(__file__).resolve().parent / "assets" / "style.css").read_text(
    encoding="utf-8"
)
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)


def _top_header() -> None:
    """BARBARIK title + tagline, with scenario id and NEW SCENARIO button."""
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="app-title">BARBARIK</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-tagline">An Intelligent Machine Learning Framework for '
            'Battlefield Simulation and Strategic Outcome Prediction</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="text-align:right;">'
            f'<span class="scenario-chip">SCENARIO {current_scenario_id()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("NEW SCENARIO", key="top_new_scenario", type="primary"):
            from session import new_scenario
            new_scenario()
            st.rerun()

    # Persistent academic-simulation banner.
    st.markdown(
        '<div class="academic-banner">ACADEMIC SIMULATION — FICTIONAL SCENARIOS ONLY</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
def _navigation() -> str:
    st.sidebar.markdown("## NAVIGATION")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "New Scenario",
            "Simulation",
            "Dataset",
            "Model Performance",
            "Explainable AI",
            "Scenario History",
            "About",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("BARBARIK · Academic ML Mini Project")
    return page


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
_top_header()
_page = _navigation()

if _page == "Dashboard":
    from pages.dashboard import render
    render()
elif _page == "New Scenario":
    from pages.scenario import render
    render()
elif _page == "Simulation":
    from pages.simulation import render
    render()
elif _page == "Dataset":
    from pages.dataset import render
    render()
elif _page == "Model Performance":
    from pages.model_performance import render
    render()
elif _page == "Explainable AI":
    from pages.explainable_ai import render
    render()
elif _page == "Scenario History":
    from pages.history import render
    render()
elif _page == "About":
    from pages.about import render
    render()