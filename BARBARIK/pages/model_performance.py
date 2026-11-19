"""
BARBARIK - Model Performance page
=================================

Shows the Random Forest configuration, train/test sizes, real classification
metrics, confusion matrix heatmap, macro-averaged ROC curves (OvR), feature
importance bar chart and the full classification report.  No numbers are
hard-coded — all come from the cached evaluate_model() computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_helpers as U

RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1,
}


def render() -> None:
    st.title("Model Performance")

    model, pipeline = U.get_model_and_pipeline()
    if model is None:
        st.error("MODEL NOT TRAINED — Please generate dataset and train model.")
        if st.button("TRAIN MODEL"):
            st.code("Run:  python train_model.py", language="bash")
        return

    metrics = U.get_model_metrics()          # real evaluate_model() output
    X_train, X_test, y_train, y_test = U.get_dataset_splits()
    _, X_test_pp, y_test_pp = U.get_transformed_splits()

    # -- RF configuration + train/test sizes --------------------------------
    st.markdown("#### Random Forest Configuration")
    st.json(RF_CONFIG)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train size", f"{X_train.shape[0]:,}")
    c2.metric("Test size", f"{X_test.shape[0]:,}")
    c3.metric("Feature count (post-encode)", f"{X_train.shape[1]}")

    st.markdown("---")

    # -- Key metrics --------------------------------------------------------
    st.markdown("#### Key Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m2.metric("Macro Precision", f"{metrics['precision_macro']:.4f}")
    m3.metric("Macro Recall", f"{metrics['recall_macro']:.4f}")
    m4.metric("Macro F1", f"{metrics['f1_macro']:.4f}")
    m5.metric("ROC-AUC (OvR)", f"{metrics['roc_auc']:.4f}"
              if metrics["roc_auc"] is not None else "n/a")

    left, right = st.columns(2)

    # -- Confusion matrix ---------------------------------------------------
    with left:
        st.markdown("#### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = list(model.classes_)
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            xgap=3, ygap=3,
        ))
        fig.update_layout(
            height=380,
            xaxis_title="Predicted", yaxis_title="Actual",
            xaxis=dict(tickangle=20, tickmode="array", tickvals=list(range(len(labels))),
                       ticktext=labels),
            yaxis=dict(tickmode="array", tickvals=list(range(len(labels))),
                       ticktext=labels),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e9f0"),
        )
        st.plotly_chart(fig, width="stretch")

    # -- ROC curves (macro OvR) ----------------------------------------------
    with right:
        st.markdown("#### ROC Curves (One-vs-Rest)")
        fig_roc = _roc_figure(model, X_test_pp, y_test_pp)
        if fig_roc is None:
            st.info("ROC curves unavailable for this model.")
        else:
            st.plotly_chart(fig_roc, width="stretch")

    st.markdown("---")

    bottom_l, bottom_r = st.columns(2)

    # -- Feature importance -------------------------------------------------
    with bottom_l:
        st.markdown("#### Feature Importance (top 15)")
        fig_imp = _importance_figure(model, pipeline)
        st.plotly_chart(fig_imp, width="stretch")

    # -- Classification report ----------------------------------------------
    with bottom_r:
        st.markdown("#### Classification Report")
        st.text(metrics["classification_report"])


def _roc_figure(model, X_test, y_test):
    """Build macro OvR ROC curves from real prediction probabilities."""
    if not hasattr(model, "predict_proba"):
        return None
    from sklearn.metrics import roc_curve, auc

    y_bin = pd.get_dummies(y_test).reindex(columns=model.classes_, fill_value=0)
    proba = model.predict_proba(X_test)

    fig = go.Figure()
    for i, cls in enumerate(model.classes_):
        fpr, tpr, _ = roc_curve(y_bin[cls], proba[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{cls} (AUC={roc_auc:.3f})",
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#8b95a7"), name="Chance",
    ))
    fig.update_layout(
        height=380,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e9f0"),
    )
    return fig


def _importance_figure(model, pipeline):
    """Real feature importances from the fitted model."""
    feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:15][::-1]  # ascending for horizontal bar
    names = [feature_names[i] for i in order]
    values = [imp[i] for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color="#3b82f6",
    ))
    fig.update_layout(
        height=480,
        xaxis_title="Importance",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e9f0"),
        yaxis=dict(autorange="reversed"),
    )
    return fig