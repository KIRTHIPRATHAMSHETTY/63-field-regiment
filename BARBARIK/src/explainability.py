"""
BARBARIK - SHAP explainability
==============================

Wraps SHAP for the fictional BARBARIK Random Forest classifier.

Important multi-class detail: for N-class classifiers ``shap.TreeExplainer``
returns a **list of arrays** (one SHAP matrix per class).  Here we always
index by the class a prediction refers to.

Robustness: every public helper wraps SHAP in a try/except and falls back to
``model.feature_importances_`` if SHAP is unavailable, printing a clear message.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    shap = None
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "SHAP unavailable, using feature_importances_ fallback."


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------
def get_shap_explainer(model, X_train_sample, n_background: int = 500):
    """Build a SHAP TreeExplainer using ``n_background`` rows as background.

    If SHAP cannot be used, returns ``None`` and logs the fallback message.
    """
    if not SHAP_AVAILABLE:
        logger.warning(FALLBACK_MESSAGE)
        return None

    background = X_train_sample[:n_background]
    try:
        return shap.TreeExplainer(model, background)
    except Exception as exc:  # pragma: no cover
        logger.warning("%s (%s)", FALLBACK_MESSAGE, exc)
        return None


def get_global_shap_values(explainer, X_test_sample):
    """Compute SHAP values for a sample of test rows.

    Returns the raw explainer output (list of arrays for multi-class).
    """
    return getattr(explainer, "shap_values")(X_test_sample)


def get_local_shap_values(explainer, X_single_instance, class_index: int = 0):
    """SHAP values for a single instance, for a specific class.

    Supports both list-of-arrays (older SHAP) and 3-D array
    ``(n_samples, n_features, n_classes)`` (newer SHAP).
    """
    shap_values_all = getattr(explainer, "shap_values")(X_single_instance)

    if isinstance(shap_values_all, list):
        arr = np.asarray(shap_values_all[class_index])
    else:
        arr = np.asarray(shap_values_all)
        # Newer SHAP: (n_samples, n_features, n_classes)
        if arr.ndim == 3:
            arr = arr[:, :, class_index]

    # Collapse to a 1-D vector for a single row.
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr, dtype=float)


def get_global_shap_values_for_class(explainer, X_sample, class_index: int = 0):
    """Return SHAP values (n_samples, n_features) for a single class."""
    shap_values_all = getattr(explainer, "shap_values")(X_sample)
    if isinstance(shap_values_all, list):
        return np.asarray(shap_values_all[class_index])
    arr = np.asarray(shap_values_all)
    if arr.ndim == 3:
        return arr[:, :, class_index]
    return arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_shap_summary(shap_values, X_sample, feature_names):
    """Summary (beeswarm) plot. Returns a matplotlib figure.

    For multi-class input, ``shap_values`` must already be reduced to the
    matrix for a single class (see caller).``
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(FALLBACK_MESSAGE)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        np.asarray(X_sample),
        feature_names=list(feature_names),
        show=False,
    )
    plt.title("SHAP summary (feature impact on outcome)")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, X_single, feature_names, class_index: int = 0):
    """Waterfall plot for a single instance. Returns a matplotlib figure.

    ``shap_values`` is the per-instance vector already reduced to one class.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(FALLBACK_MESSAGE)

    explainer_shap = shap.Explanation(
        np.asarray(shap_values, dtype=float),
        base_values=np.asarray(shap_values, dtype=float).mean(),
        data=np.asarray(X_single)[0],
        feature_names=list(feature_names),
    )
    fig = shap.plots.waterfall(explainer_shap, show=False, max_display=12)
    return fig


# ---------------------------------------------------------------------------
# Top features
# ---------------------------------------------------------------------------
def get_top_influencing_features(shap_values_local, feature_names, top_n: int = 8):
    """Return the top-|SHAP| features for a single instance.

    Returns a list of (feature_name, signed_shap_value) sorted by absolute
    SHAP value descending.
    """
    sv = np.asarray(shap_values_local, dtype=float).reshape(-1)
    names = list(feature_names)[: sv.size]

    pairs = sorted(zip(names, sv), key=lambda t: abs(t[1]), reverse=True)
    return pairs[:top_n]


if __name__ == "__main__":
    # Optional quick self-check so the module is importable and pure.
    pass