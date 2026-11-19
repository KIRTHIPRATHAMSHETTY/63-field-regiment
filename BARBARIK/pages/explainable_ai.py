"""
BARBARIK - Explainable AI page
==============================

SHAP-based explanations: a summary (beeswarm) plot for a chosen class, a
waterfall for the current scenario, and a positive/negative contribution list.
Shows the real outputs of src/explainability.py — never fabricated. Falls back
to a clear message if SHAP is unavailable.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui_helpers as U


def render() -> None:
    st.title("Explainable AI")

    model, _ = U.get_model_and_pipeline()
    if model is None:
        st.error("MODEL NOT TRAINED — Please generate dataset and train model.")
        if st.button("TRAIN MODEL"):
            st.code("Run:  python train_model.py", language="bash")
        return

    explainer = U.get_shap_explainer_cached()
    if explainer is None:
        st.warning("SHAP unavailable — running in feature_importances_ fallback mode.")
        _fallback_importance(model)
        return

    st.markdown("#### SHAP Summary (feature impact on predicted outcome)")
    class_choice = st.selectbox(
        "Explain class",
        list(model.classes_),
    )
    class_idx = int(list(model.classes_).index(class_choice))

    try:
        shap_values = U.get_global_shap_cached()
        matrix = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not compute global SHAP values: {exc}")
        return

    from explainability import plot_shap_summary
    feature_names = U.get_feature_names()
    _, X_test_pp, _ = U.get_transformed_splits()
    X_sample = X_test_pp[:matrix.shape[0]]

    fig = plot_shap_summary(matrix, X_sample, feature_names)
    st.pyplot(fig)

    st.markdown("---")

    # -- Current scenario local explanation --------------------------------
    st.markdown("#### Current Scenario — Local Explanation")
    left, right = st.columns(2)

    with left:
        st.markdown("**Positive / negative feature contributions**")
        predicted, local_sv, top, _X = U.get_local_shap_for_scenario(st.session_state)

        if predicted is None:
            st.info("Run a scenario first to see local attributions.")
        else:
            st.caption(f"Explaining predicted class: {predicted.replace('_', ' ')}")
            if top:
                pos = [(n, v) for n, v in top if v > 0]
                neg = [(n, v) for n, v in top if v < 0]

                st.markdown("**Pushes toward this class (+)**:")
                if pos:
                    for n, v in pos:
                        st.markdown(f"· <span style='color:#3b82f6'>{n}</span> "
                                    f"<b>+{v:.4f}</b>", unsafe_allow_html=True)
                else:
                    st.caption("None in top-8.")

                st.markdown("**Against this class (−)**:")
                if neg:
                    for n, v in neg:
                        st.markdown(f"· <span style='color:#ef4444'>{n}</span> "
                                    f"<b>{v:.4f}</b>", unsafe_allow_html=True)
                else:
                    st.caption("None in top-8.")
            else:
                st.caption("No local SHAP contributions computed.")

    with right:
        st.markdown("**Waterfall — current scenario**")
        if predicted is None or local_sv is None or _X is None:
            st.info("Run a scenario first to see the waterfall.")
        else:
            try:
                from explainability import plot_shap_waterfall
                wf = plot_shap_waterfall(local_sv, _X,
                                         feature_names, class_index=0)
                st.pyplot(wf)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Waterfall plot unavailable: {exc}")


def _fallback_importance(model) -> None:
    """feature_importances_ fallback shown when SHAP is unavailable."""
    pipeline = _pipeline_or_none()
    names = list(pipeline.get_feature_names_out()) if pipeline else [
        f"f{i}" for i in range(len(model.feature_importances_))
    ]
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:10][::-1]
    fig = go.Figure(go.Bar(
        x=[imp[i] for i in order],
        y=[names[i] for i in order],
        orientation="h", marker_color="#3b82f6",
    ))
    fig.update_layout(title="Top 10 (feature_importances_ fallback)",
                      height=420, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e9f0"))
    st.plotly_chart(fig, width="stretch")
    st.caption("SHAP unavailable, using feature_importances_ fallback.")


def _pipeline_or_none():
    _, pipeline = U.get_model_and_pipeline()
    return pipeline