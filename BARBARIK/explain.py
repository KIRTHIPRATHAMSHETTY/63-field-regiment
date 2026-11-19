"""
BARBARIK - SHAP explainability demo
===================================

Loads the saved model + pipeline, takes one row from the test set, predicts
its outcome, computes local SHAP values for the predicted class, prints the
top-8 most influential features and saves a SHAP summary plot PNG.

Run:  python explain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from explainability import (  # noqa: E402
    get_global_shap_values,
    get_local_shap_values,
    get_shap_explainer,
    get_top_influencing_features,
    plot_shap_summary,
)
from model import load_model  # noqa: E402
from preprocessing import load_dataset, split_data  # noqa: E402

ASSET_DIR = Path("assets/images")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset + split (same scheme as training, so repro matches).
    df = load_dataset("data/barbarik_dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Load the trained model and its preprocessing pipeline.
    model, pipeline = load_model("models/")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    X_train_pp = pipeline.transform(X_train)
    X_test_pp = pipeline.transform(X_test)

    # --- SHAP explainer (500-row background) ------------------------------
    explainer = get_shap_explainer(model, X_train_pp, n_background=500)
    if explainer is None:
        print("FALLBACK: SHAP unavailable, using feature_importances_ instead.")
        imp = model.feature_importances_
        pairs = sorted(
            zip(feature_names, imp), key=lambda t: t[1], reverse=True
        )[:8]
        print("\nTop 8 features (feature_importances_ fallback):")
        for name, val in pairs:
            print(f"  {name:30s} {val:.5f}")
        return

    # --- Pick one test row, predict, explain ------------------------------
    index = 0
    X_row_pp = X_test_pp[index : index + 1]
    y_true = y_test.iloc[index]

    class_idx_pred = int(np.argmax(model.predict_proba(X_row_pp)[0]))
    classes = list(model.classes_)
    predicted_class = classes[class_idx_pred]

    print(f"Test row index: {index}")
    print(f"True label     : {y_true}")
    print(f"Predicted class: {predicted_class} (class_index={class_idx_pred})")

    # --- Global SHAP on a test sample, and a summary plot -----------------
    X_sample = X_test_pp[:300]
    global_sv = get_global_shap_values(explainer, X_sample)

    # summary plot is per-class -> use the predicted class matrix.
    class_matrix = (
        global_sv[class_idx_pred] if isinstance(global_sv, list) else global_sv
    )
    summary_fig = plot_shap_summary(class_matrix, X_sample, feature_names)
    summary_path = ASSET_DIR / "shap_summary.png"
    summary_fig.savefig(summary_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved summary plot: {summary_path}")

    # --- Local SHAP for the predicted class -------------------------------
    local_sv = get_local_shap_values(explainer, X_row_pp, class_index=class_idx_pred)

    top = get_top_influencing_features(local_sv, feature_names, top_n=8)
    print(f"\nTop {len(top)} influencing features for '{predicted_class}' "
          f"(signed SHAP values):")
    for name, val in top:
        arrow = "+" if val >= 0 else "-"
        print(f"  {arrow} {name:30s} {val:+.5f}")

    # Show the raw scenario too, for context.
    row_raw = X_test.iloc[index]
    print("\nRaw scenario features (first 8):")
    for col, val in row_raw.head(8).items():
        print(f"  {col:28s} {val}")


if __name__ == "__main__":
    main()