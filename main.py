from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import os
from datetime import datetime

from src.simulation_engine import simulate_scenario
from src.dataset_generator import generate_dataset
from src.model import OutcomePredictor
from src.explainability import ModelExplainer
from src.database import ScenarioDatabase
from src.assistant import generate_tactical_reply


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    scenario: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None


class ScenarioRequest(BaseModel):
    side_a_force_strength: int
    side_b_force_strength: int
    side_a_mobility: float
    side_b_mobility: float
    side_a_logistics: float
    side_b_logistics: float
    side_a_intelligence: float
    side_b_intelligence: float
    side_a_communication: float
    side_b_communication: float
    side_a_readiness: float
    side_b_readiness: float
    side_a_morale: float
    side_b_morale: float
    side_a_training: float
    side_b_training: float
    side_a_supply_distance: int
    side_b_supply_distance: int
    side_a_surprise: float
    side_b_surprise: float
    side_a_fatigue: float
    side_b_fatigue: float
    side_a_equipment_quality: float
    side_b_equipment_quality: float
    electronic_warfare_level: float
    objective_type: str
    season: str
    rules_of_engagement: str
    terrain_type: str
    weather: str
    visibility: str
    time_of_day: str
    environmental_difficulty: float


# Initialize FastAPI app
app = FastAPI(title="BARBARIK - ML-Based Wargaming Dashboard")

# Global variables
predictor: Optional[OutcomePredictor] = None
explainer: Optional[ModelExplainer] = None
db = ScenarioDatabase()
current_scenario_id = "SCN-2024-0356"
scenario_counter = 356


def initialize_model():
    """Initialize or load the ML model."""
    global predictor, explainer
    
    model_path = 'models/xgboost_model.joblib'
    if not os.path.exists(model_path):
        model_path = 'models/random_forest_model.joblib'
    preprocessor_path = 'models/preprocessing_pipeline.joblib'
    label_encoder_path = 'models/label_encoder.joblib'
    dataset_path = 'data/barbarik_dataset_new.csv'
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        print(f"Loading existing model from {model_path}...")
        predictor = OutcomePredictor()
        predictor.load(model_path, preprocessor_path, label_encoder_path)
        
        # Initialize explainer with background data
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            explainer = ModelExplainer(predictor)
            explainer.initialize_explainer(df, n_background=100)
    else:
        print("Model not found. Training new XGBoost model...")
        # Generate dataset if needed
        if not os.path.exists(dataset_path):
            print("Generating dataset...")
            generate_dataset(20000, dataset_path)
        
        # Train model
        df = pd.read_csv(dataset_path)
        predictor = OutcomePredictor()
        predictor.train(df)
        predictor.save(model_path, preprocessor_path, label_encoder_path)
        
        # Initialize explainer
        explainer = ModelExplainer(predictor)
        explainer.initialize_explainer(df, n_background=100)


# Initialize on startup
@app.on_event("startup")
def startup_event():
    initialize_model()


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/dataset/stats")
async def get_dataset_stats():
    """Get dataset summary statistics."""
    dataset_path = 'data/barbarik_dataset_new.csv'
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = pd.read_csv(dataset_path)
    
    return {
        "total_scenarios": len(df),
        "class_distribution": df['outcome'].value_counts(normalize=True).to_dict(),
        "feature_ranges": {
            "force_strength": {
                "min": int(df[['side_a_force_strength', 'side_b_force_strength']].min().min()),
                "max": int(df[['side_a_force_strength', 'side_b_force_strength']].max().max())
            }
        }
    }


@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get model performance metrics."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    # Re-calculate metrics on test set
    dataset_path = 'data/barbarik_dataset_new.csv'
    df = pd.read_csv(dataset_path)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    X = df.drop(columns=['scenario_id', 'outcome'], errors='ignore')
    y = df['outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess
    X_test_processed = predictor.preprocessor.transform(X_test)
    
    # Predict
    y_pred_encoded = predictor.model.predict(X_test_processed)
    y_pred = predictor.label_encoder.inverse_transform(y_pred_encoded)
    y_pred_proba = predictor.model.predict_proba(X_test_processed)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    classes = predictor.label_encoder.classes_
    try:
        y_test_encoded = predictor.label_encoder.transform(y_test)
        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # ROC curve data
    roc_curve_data = {}
    y_test_bin = label_binarize(y_test, classes=classes)
    
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_curve_data[str(class_name)] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    # Feature importances
    feature_names = predictor.preprocessor.numeric_features + list(
        predictor.preprocessor.preprocessor.named_transformers_['cat'].get_feature_names_out(
            predictor.preprocessor.categorical_features
        )
    )
    importances = dict(zip(feature_names, [float(v) for v in predictor.model.feature_importances_]))
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "roc_curve_data": roc_curve_data,
        "feature_importances": importances
    }


@app.post("/api/predict")
async def predict_scenario(scenario: ScenarioRequest):
    """Predict outcome for a given scenario."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    # Convert to dict
    scenario_dict = scenario.model_dump() if hasattr(scenario, 'model_dump') else scenario.dict()
    scenario_dict['scenario_id'] = current_scenario_id
    scenario_dict['outcome'] = 'STALEMATE'  # Dummy, will be dropped
    
    # Calculate the 3 new derived features
    scenario_dict['force_ratio'] = float(scenario_dict['side_a_force_strength']) / (float(scenario_dict['side_b_force_strength']) + 1.0)
    scenario_dict['intel_diff'] = float(scenario_dict['side_a_intelligence']) - float(scenario_dict['side_b_intelligence'])
    scenario_dict['logistics_diff'] = float(scenario_dict['side_a_logistics']) - float(scenario_dict['side_b_logistics'])
    
    # Run simulation
    sim_result = simulate_scenario(scenario_dict)
    
    # Get ML prediction
    prediction = predictor.predict_outcome(scenario_dict)
    
    # Get SHAP values
    shap_features = []
    if explainer is not None:
        top_features = explainer.get_top_influencing_features(scenario_dict, top_n=8)
        shap_features = [
            {"feature": f, "value": float(v)} for f, v in top_features
        ]
    
    # Save to database
    side_a = {
        'force_strength': scenario.side_a_force_strength,
        'mobility': scenario.side_a_mobility,
        'logistics': scenario.side_a_logistics,
        'intelligence': scenario.side_a_intelligence,
        'communication': scenario.side_a_communication,
        'readiness': scenario.side_a_readiness,
        'morale': scenario.side_a_morale,
        'training': scenario.side_a_training,
        'supply_distance': scenario.side_a_supply_distance,
        'surprise': scenario.side_a_surprise,
        'fatigue': scenario.side_a_fatigue,
        'equipment_quality': scenario.side_a_equipment_quality
    }
    
    side_b = {
        'force_strength': scenario.side_b_force_strength,
        'mobility': scenario.side_b_mobility,
        'logistics': scenario.side_b_logistics,
        'intelligence': scenario.side_b_intelligence,
        'communication': scenario.side_b_communication,
        'readiness': scenario.side_b_readiness,
        'morale': scenario.side_b_morale,
        'training': scenario.side_b_training,
        'supply_distance': scenario.side_b_supply_distance,
        'surprise': scenario.side_b_surprise,
        'fatigue': scenario.side_b_fatigue,
        'equipment_quality': scenario.side_b_equipment_quality
    }
    
    environment = {
        'electronic_warfare_level': scenario.electronic_warfare_level,
        'objective_type': scenario.objective_type,
        'season': scenario.season,
        'rules_of_engagement': scenario.rules_of_engagement,
        'terrain_type': scenario.terrain_type,
        'weather': scenario.weather,
        'visibility': scenario.visibility,
        'time_of_day': scenario.time_of_day,
        'environmental_difficulty': scenario.environmental_difficulty
    }
    
    db.save_scenario(
        current_scenario_id,
        side_a,
        side_b,
        environment,
        prediction['predicted_class'],
        prediction['probabilities']
    )
    
    # Format outcome for display
    if prediction['predicted_class'] == 'SIDE_A_ADVANTAGE':
        outcome_display = 'BLUE FORCE ADVANTAGE'
    elif prediction['predicted_class'] == 'SIDE_B_ADVANTAGE':
        outcome_display = 'RED FORCE ADVANTAGE'
    else:
        outcome_display = 'STALEMATE'
    
    # Calculate confidence
    probs = prediction['probabilities']
    confidence = max(probs.values())
    
    # Calculate estimated casualties based on simulation scores
    score_a = sim_result['simulation_score_a']
    score_b = sim_result['simulation_score_b']
    
    # Casualties inversely related to score
    base_casualties = 500
    casualties_blue = int(base_casualties * (1 - score_a) * np.random.uniform(0.8, 1.2))
    casualties_red = int(base_casualties * (1 - score_b) * np.random.uniform(0.8, 1.2))
    
    return {
        "outcome": outcome_display,
        "predicted_class": prediction['predicted_class'],
        "probabilities": {
            "SIDE_A_ADVANTAGE": float(probs['SIDE_A_ADVANTAGE']),
            "SIDE_B_ADVANTAGE": float(probs['SIDE_B_ADVANTAGE']),
            "STALEMATE": float(probs['STALEMATE'])
        },
        "confidence": float(confidence),
        "simulation_metrics": {
            "duration": sim_result['duration'],
            "score_a": sim_result['simulation_score_a'],
            "score_b": sim_result['simulation_score_b'],
            "resource_status_a": sim_result['resource_status_a'],
            "resource_status_b": sim_result['resource_status_b'],
            "casualties_blue": f"{casualties_blue - 50}-{casualties_blue + 50}",
            "casualties_red": f"{casualties_red - 50}-{casualties_red + 50}"
        },
        "shap_values": shap_features
    }


@app.get("/api/history")
async def get_history():
    """Get recent scenario history."""
    history = db.get_history(limit=20)
    return {"history": history}


@app.post("/api/chat")
async def chat_with_assistant(req: ChatRequest):
    """Chat with the tactical AI copilot and viva advisor."""
    reply_data = generate_tactical_reply(
        message=req.message,
        scenario=req.scenario,
        prediction=req.prediction
    )
    return reply_data


@app.post("/api/new-scenario")
async def new_scenario():
    """Generate a new scenario ID."""
    global current_scenario_id, scenario_counter
    
    scenario_counter += 1
    year = datetime.now().year
    current_scenario_id = f"SCN-{year}-{scenario_counter:04d}"
    
    return {"scenario_id": current_scenario_id}


@app.post("/api/retrain")
async def retrain_model():
    """Regenerate dataset and retrain model."""
    global predictor, explainer
    
    # Regenerate dataset
    print("Regenerating dataset...")
    generate_dataset(20000, 'data/barbarik_dataset_new.csv')
    
    # Retrain model
    print("Retraining model...")
    df = pd.read_csv('data/barbarik_dataset_new.csv')
    predictor = OutcomePredictor()
    metrics = predictor.train(df)
    predictor.save('models/xgboost_model.joblib', 'models/preprocessing_pipeline.joblib', 'models/label_encoder.joblib')
    import joblib
    joblib.dump(predictor.model, 'models/random_forest_model.joblib')
    
    # Reinitialize explainer
    explainer = ModelExplainer(predictor)
    explainer.initialize_explainer(df, n_background=100)
    
    return {
        "message": "Model retrained successfully",
        "metrics": {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "roc_auc": metrics['roc_auc']
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
