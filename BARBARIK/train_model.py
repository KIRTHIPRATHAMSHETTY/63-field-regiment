"""
BARBARIK - End-to-end model training / evaluation
=================================================

Loads the generated dataset, splits it (stratified, 80/20), fits a
preprocessing pipeline on the TRAIN split only (no leakage), trains a
Random Forest, evaluates on the held-out test split, saves model + pipeline
and prints all metrics (computed via real sklearn calls).

Run:  python train_model.py
"""

from __future__ import annotations

import time

import numpy as np

from src.model import (
    evaluate_model,
    save_model,
    train_model,
)
from src.preprocessing import (
    build_preprocessing_pipeline,
    load_dataset,
    split_data,
)

DATASET_PATH = "data/barbarik_dataset.csv"


def main() -> None:
    t0 = time.time()

    # 1. Load and split (stratified 80/20, random_state=42).
    df = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Fit the preprocessing pipeline on TRAIN ONLY (no leakage).
    pipeline = build_preprocessing_pipeline()
    X_train_pp = pipeline.fit_transform(X_train, y_train)
    X_test_pp = pipeline.transform(X_test)

    # 3. Train the Random Forest on preprocessed train features.
    model = train_model(X_train_pp, y_train)

    # 4. Evaluate on the held-out test set (real sklearn metrics only).
    metrics = evaluate_model(model, X_test_pp, y_test)

    # 5. Persist model + pipeline.
    save_model(model, pipeline, model_dir="models/")

    # ---- Reporting -----------------------------------------------------
    print(f"Trained + evaluated in {time.time() - t0:.1f}s")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Macro Precision    : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall       : {metrics['recall_macro']:.4f}")
    print(f"Macro F1           : {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC (macro,ovr): {metrics['roc_auc']:.4f}\n")

    print("Classification report:")
    print(metrics["classification_report"])

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(metrics["confusion_matrix"])

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    print("\nTop 10 feature importances:")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(order, start=1):
        print(f"  {rank:2d}. {feature_names[idx]:30s} {importances[idx]:.5f}")

    print("\nSaved to models/: random_forest_model.joblib, preprocessing_pipeline.joblib")


if __name__ == "__main__":
    main()