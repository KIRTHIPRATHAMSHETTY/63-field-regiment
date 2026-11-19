"""
BARBARIK - FastAPI backend
==========================

Serves the static dashboard and exposes JSON API endpoints for dataset stats,
model metrics, scenario prediction (with SHAP), scenario history, and retrain.

Run:  python -m uvicorn main:app --reload
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from dataset_generator import generate_dataset
from model import (
    evaluate_model,
    load_model,
    predict_outcome,
    save_model,
    train_model,
)
from preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessing_pipeline,
    load_dataset,
    split_data,
)
from simulation_engine import simulate_scenario, build_default_config
from database import get_all_scenarios, init_db, save_scenario
from explainability import (
    get_global_shap_values_for_class,
    get_local_shap_values,
    get_shap_explainer,
    get_top_influencing_features,
)

DATASET_PATH = Path("data/barbarik_dataset.csv")
MODEL_DIR = Path("models")

app = FastAPI(title="BARBARIK")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory runtime state (model + pipeline + cached computations).
_runtime: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ScenarioPayload(BaseModel):
    scenario: dict[str, Any]


class RetrainPayload(BaseModel):
    n_scenarios: int = 20000


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------
def ensure_dataset() -> None:
    if not DATASET_PATH.exists():
        generate_dataset(20000, output_path=str(DATASET_PATH))


def ensure_model() -> None:
    model_file = MODEL_DIR / "random_forest_model.joblib"
    if not model_file.exists():
        df = load_dataset(str(DATASET_PATH))
        X_train, X_test, y_train, y_test = split_data(df)
        pipeline = build_preprocessing_pipeline()
        X_train_pp = pipeline.fit_transform(X_train, y_train)
        model = train_model(X_train_pp, y_train)
        save_model(model, pipeline, model_dir=str(MODEL_DIR))


def load_runtime() -> None:
    model, pipeline = load_model(str(MODEL_DIR))
    df = load_dataset(str(DATASET_PATH))
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_pp = pipeline.transform(X_train)
    X_test_pp = pipeline.transform(X_test)
    feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    explainer = get_shap_explainer(model, X_train_pp, n_background=300)

    _runtime.update(
        model=model,
        pipeline=pipeline,
        X_test=X_test,
        X_test_pp=X_test_pp,
        y_test=y_test,
        feature_names=feature_names,
        explainer=explainer,
    )


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_dataset()
    ensure_model()
    load_runtime()


# ---------------------------------------------------------------------------
# Static page
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# API: dataset stats
# ---------------------------------------------------------------------------
@app.get("/api/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    df = load_dataset(str(DATASET_PATH))
    counts = df["outcome"].value_counts().to_dict()
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "missing_values": int(df.isna().sum().sum()),
    }


# ---------------------------------------------------------------------------
# API: model metrics
# ---------------------------------------------------------------------------
@app.get("/api/model/metrics")
def model_metrics() -> dict[str, Any]:
    model = _runtime["model"]
    X_test_pp = _runtime["X_test_pp"]
    y_test = _runtime["y_test"]
    pipeline = _runtime["pipeline"]
    feature_names = _runtime["feature_names"]

    metrics = evaluate_model(model, X_test_pp, y_test)

    # --- ROC curve (macro one-vs-rest) -----------------------------------
    from sklearn.metrics import roc_curve

    y_bin = np.array([1 if y == "SIDE_A_ADVANTAGE" else 0 for y in y_test])
    proba = model.predict_proba(X_test_pp)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    a_idx = class_to_idx.get("SIDE_A_ADVANTAGE", 0)
    fpr, tpr, _ = roc_curve(y_bin, proba[:, a_idx])

    # --- Feature importances ---------------------------------------------
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    top_importances = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in order
    ]

    return {
        "model_used": "Random Forest Classifier",
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision_macro"], 4),
        "recall": round(metrics["recall_macro"], 4),
        "f1": round(metrics["f1_macro"], 4),
        "roc_auc": round(metrics["roc_auc"], 4) if metrics["roc_auc"] else None,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_labels": list(model.classes_),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(metrics["roc_auc"]), 4) if metrics["roc_auc"] else None,
        },
        "feature_importances": top_importances,
        "train_size": 16000,
        "test_size": 4000,
    }


# ---------------------------------------------------------------------------
# API: predict
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict(payload: ScenarioPayload) -> dict[str, Any]:
    model = _runtime["model"]
    pipeline = _runtime["pipeline"]
    explainer = _runtime.get("explainer")
    feature_names = _runtime["feature_names"]

    scenario = payload.scenario

    # --- Simulation metrics (real) ---------------------------------------
    sim = simulate_scenario(scenario)

    # --- ML prediction (real) --------------------------------------------
    pred = predict_outcome(model, pipeline, scenario)
    predicted = pred["predicted_class"]
    probs = pred["probabilities"]

    # --- Local SHAP -------------------------------------------------------
    X_single = pipeline.transform(_to_df(scenario))
    shap_top = []
    if explainer is not None:
        try:
            class_idx = int(list(model.classes_).index(predicted))
            sv = get_local_shap_values(explainer, X_single, class_index=class_idx)
            shap_top = [
                {"feature": n, "value": round(float(v), 5)}
                for n, v in get_top_influencing_features(sv, feature_names, top_n=8)
            ]
        except Exception:
            shap_top = []

    # --- Persist to SQLite -----------------------------------------------
    sid = scenario.get("scenario_id") or f"SCN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    side_a = {k: scenario[k] for k in scenario if k.startswith("side_a_")}
    side_b = {k: scenario[k] for k in scenario if k.startswith("side_b_")}
    environment = {
        k: scenario[k]
        for k in ("terrain_type", "weather", "visibility", "time_of_day",
                  "environmental_difficulty", "electronic_warfare_level",
                  "objective_type")
        if k in scenario
    }
    save_scenario(
        scenario_id=sid,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        side_a=side_a,
        side_b=side_b,
        environment=environment,
        predicted_outcome=predicted,
        prob_a=probs.get("SIDE_A_ADVANTAGE", 0.0),
        prob_b=probs.get("SIDE_B_ADVANTAGE", 0.0),
        prob_draw=probs.get("STALEMATE", 0.0),
    )

    return {
        "scenario_id": sid,
        "outcome": predicted,
        "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        "simulation_metrics": {
            "duration": sim["duration"],
            "score_a": round(float(sim["simulation_score_a"]), 4),
            "score_b": round(float(sim["simulation_score_b"]), 4),
            "resource_a": sim["resource_status_a"],
            "resource_b": sim["resource_status_b"],
        },
        "shap_values": shap_top,
    }


def _to_df(scenario: dict[str, Any]):
    import pandas as pd
    return pd.DataFrame([{c: scenario.get(c, 0) for c in FEATURE_COLUMNS}])


# ---------------------------------------------------------------------------
# API: history
# ---------------------------------------------------------------------------
@app.get("/api/history")
def history(limit: int = 20) -> dict[str, Any]:
    rows = get_all_scenarios()[:limit]
    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "prob_a": r["prob_a"],
                "prob_b": r["prob_b"],
                "prob_draw": r["prob_draw"],
                "predicted_outcome": r["predicted_outcome"],
            }
        )
    return {"history": items}


# ---------------------------------------------------------------------------
# API: new scenario
# ---------------------------------------------------------------------------
@app.post("/api/new-scenario")
def new_scenario() -> dict[str, Any]:
    sid = f"SCN-{datetime.now().strftime('%Y-%m-%d')}-{np.random.randint(1000,9999)}"
    return {"scenario_id": sid, "defaults": build_default_config()}


# ---------------------------------------------------------------------------
# API: retrain
# ---------------------------------------------------------------------------
@app.post("/api/retrain")
def retrain(payload: RetrainPayload) -> dict[str, Any]:
    generate_dataset(payload.n_scenarios, output_path=str(DATASET_PATH))
    df = load_dataset(str(DATASET_PATH))
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_preprocessing_pipeline()
    X_train_pp = pipeline.fit_transform(X_train, y_train)
    X_test_pp = pipeline.transform(X_test)
    model = train_model(X_train_pp, y_train)
    save_model(model, pipeline, model_dir=str(MODEL_DIR))
    metrics = evaluate_model(model, X_test_pp, y_test)
    load_runtime()
    return {
        "status": "ok",
        "n_scenarios": payload.n_scenarios,
        "accuracy": round(metrics["accuracy"], 4),
        "f1": round(metrics["f1_macro"], 4),
    }


# ---------------------------------------------------------------------------
# API: global SHAP (for beeswarm plot)
# ---------------------------------------------------------------------------
@app.get("/api/shap/summary")
def shap_summary(sample_size: int = 50) -> dict[str, Any]:
    model = _runtime["model"]
    X_test_pp = _runtime["X_test_pp"]
    explainer = _runtime.get("explainer")
    feature_names = _runtime["feature_names"]

    if explainer is None:
        return {"available": False, "features": [], "values": []}

    n = min(sample_size, X_test_pp.shape[0])
    X_sample = X_test_pp[:n]
    try:
        class_idx = int(list(model.classes_).index("SIDE_A_ADVANTAGE"))
        sv = get_global_shap_values_for_class(explainer, X_sample, class_index=class_idx)
        sv = np.asarray(sv)

        # Return top-8 features by mean |SHAP|
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:8]
        features = [feature_names[i] for i in top_idx]
        values = sv[:, top_idx].tolist()
        feature_values = X_sample[:, top_idx].tolist()
        return {
            "available": True,
            "features": features,
            "shap_values": values,
            "feature_values": feature_values,
        }
    except Exception:
        return {"available": False, "features": [], "values": []}
