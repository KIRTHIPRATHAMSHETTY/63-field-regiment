# BARBARIK — Intelligent Wargaming & Outcome Prediction Framework

BARBARIK is an advanced machine-learning framework designed for tactical military wargaming outcome prediction. Using simulated bilateral combat scenarios, the system predicts strategic outcomes (Blue Advantage, Red Advantage, or Stalemate) with high accuracy and provides full decision interpretability via Explainable AI (SHAP).

---

## 🚀 Key Specifications & Architecture

### 1. Primary Algorithm: XGBoost (Extreme Gradient Boosting)
- **Model**: `XGBClassifier` (`xgboost` library)
- **Architecture**: Gradient-boosted decision tree ensemble optimized for tabular military engagement parameters.
- **Measured Test-Set Accuracy**: **90.35%** on a held-out stratified 20% test split (20,000 scenarios total).
- **ROC-AUC (Multiclass OvR)**: **0.98161**
- **Weighted F1-Score**: **0.90334**
- **Target Encoding**: Supervised `LabelEncoder` managing class mappings (`SIDE_A_ADVANTAGE`, `SIDE_B_ADVANTAGE`, `STALEMATE`).

### 2. Dataset & Features (36 Total Features)
The simulation dataset consists of 20,000 scenarios generated using a deterministic simulation engine (seed `42`, Gaussian noise standard deviation $\sigma = 0.015$).

The input feature space consists of **36 features** (29 numerical, 7 categorical):
- **Force Capabilities (Side A & Side B - 12 features)**: Force Strength (`30-100`), Mobility (`0.3-1.0`), Logistics (`0.3-1.0`), Intelligence (`0.3-1.0`), Communication (`0.3-1.0`), Readiness (`0.3-1.0`).
- **Tactical Modifiers (Side A & Side B - 10 features)**: Morale (`0.0-1.0`), Training (`0.0-1.0`), Surprise (`0.0-1.0`), Fatigue (`0.0-1.0`), Equipment Quality (`0.3-1.0`).
- **Logistics Geography (Side A & Side B - 2 features)**: Supply Distance (`0-500 km`).
- **Environment & Mission (7 categorical, 2 numerical)**:
  - Categorical: `terrain_type`, `weather`, `visibility`, `time_of_day`, `objective_type`, `season`, `rules_of_engagement`.
  - Numerical: `electronic_warfare_level`, `environmental_difficulty`.
- **Derived Features (3 features)**:
  - `force_ratio = side_a_force_strength / (side_b_force_strength + 1)`
  - `intel_diff = side_a_intelligence - side_b_intelligence`
  - `logistics_diff = side_a_logistics - side_b_logistics`

### 3. Explainable AI (XAI)
- **Methodology**: Tree-based Shapley Additive exPlanations (`shap.TreeExplainer`).
- **Capabilities**: Computes exact local feature attribution values per scenario prediction to explain *why* an advantage or stalemate was predicted, as well as global feature rankings.

### 4. Technology Stack
- **Backend**: FastAPI, Uvicorn (ASGI server), Python 3.13
- **Machine Learning**: XGBoost, scikit-learn, SHAP, NumPy, Pandas, Joblib
- **Database**: SQLite (`scenarios.db`) for tracking simulation history
- **Frontend**: Responsive military tactical HUD (HTML5, CSS3, Vanilla JavaScript, Plotly.js for interactive radar/ROC charts). The layout remains an authentic command console interface, untouched and preserved during this ML pipeline upgrade.

---

## 📊 Measured Performance Metrics

| Metric | Measured Value |
|---|---|
| **Test Accuracy** | **90.35%** |
| **Weighted Precision** | **0.90541** |
| **Weighted Recall** | **0.90350** |
| **Weighted F1-Score** | **0.90334** |
| **Macro F1-Score** | **0.90181** |
| **Multiclass ROC-AUC** | **0.98161** |

### Confusion Matrix (Test Set: 4,000 scenarios):
| True \ Predicted | Blue Advantage | Red Advantage | Stalemate |
|---|---|---|---|
| **Blue Advantage** | 856 | 0 | 122 |
| **Red Advantage** | 0 | 856 | 141 |
| **Stalemate** | 52 | 71 | 1902 |

---

## 🛠️ Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train XGBoost Model
```bash
python train_model.py
```

### 3. Run Automated Tests
```bash
python tests/test_api_and_model.py
```

### 4. Launch Application Server
```bash
python -m uvicorn main:app --port 8000
```
Open **`http://localhost:8000`** in any modern web browser.
