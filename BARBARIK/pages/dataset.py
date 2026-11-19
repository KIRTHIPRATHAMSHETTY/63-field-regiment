"""
BARBARIK - Dataset page
=======================

Shows dataset statistics (shape, class distribution, missing values) and
Plotly visualisations (outcome distribution, correlation matrix, feature
distributions). Includes a DOWNLOAD DATASET button. All figures are computed
from the real loaded dataset.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui_helpers as U


def render() -> None:
    st.title("Dataset")
    df = U.get_dataset()  # cached, real load

    # -- Basic stats --------------------------------------------------------
    st.markdown("#### Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Classes", f"{df['outcome'].nunique()}")
    missing = int(df.isna().sum().sum())
    c4.metric("Missing Values", f"{missing}")

    st.markdown("---")
    left, right = st.columns(2)

    # -- Outcome distribution ----------------------------------------------
    with left:
        st.markdown("#### Outcome Distribution")
        counts = df["outcome"].value_counts()
        fig = px.bar(
            counts,
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map={
                "SIDE_A_ADVANTAGE": "#3b82f6",
                "SIDE_B_ADVANTAGE": "#ef4444",
                "STALEMATE": "#10b981",
            },
        )
        fig.update_layout(height=320, xaxis_title="Outcome",
                          yaxis_title="Count", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e5e9f0"))
        st.plotly_chart(fig, width="stretch")

    # -- Correlation matrix (numeric only) ----------------------------------
    with right:
        st.markdown("#### Correlation Matrix (numeric features)")
        numeric = df.select_dtypes(include="number")
        corr = numeric.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e5e9f0"), coloraxis_showscale=True)
        st.plotly_chart(fig, width="stretch")

    # -- Feature distributions ----------------------------------------------
    st.markdown("#### Feature Distributions")
    dist_cols = [
        "side_a_force_strength", "side_b_force_strength",
        "side_a_intelligence", "side_b_intelligence",
        "side_a_mobility", "side_b_mobility",
    ]
    selected = st.selectbox("Feature", dist_cols)
    hist = px.histogram(df, x=selected, nbins=40,
                        color_discrete_sequence=["#3b82f6"])
    hist.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e9f0"))
    st.plotly_chart(hist, width="stretch")

    st.markdown("---")
    st.markdown("#### Raw Data Preview")
    st.dataframe(df.head(20), width="stretch")

    # -- Download -----------------------------------------------------------
    st.download_button(
        "DOWNLOAD DATASET (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="barbarik_dataset.csv",
        mime="text/csv",
    )