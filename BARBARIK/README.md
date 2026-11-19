# BARBARIK

_An Intelligent Machine Learning Framework for Battlefield Simulation and Strategic Outcome Prediction_ (Academic AIML Mini-Project)

> **ACADEMIC SIMULATION — FICTIONAL SCENARIOS ONLY.** This project uses no real
> military data, no real targets, and no real borders. Every scenario, force and
> outcome is abstract and for educational purposes only.

---

## 1. Project Overview

BARBARIK couples a **stochastic, rule-based wargaming simulation engine** with a
**machine-learning outcome predictor**. A user defines a scenario — two fictional
sides with force, mobility, logistics, intelligence, communication and readiness
scores, plus a terrain, weather, visibility and time-of-day — and the system:

1. runs a stochastic simulation to produce an abstract outcome, and
2. trains a Random Forest to **predict that outcome directly from the raw inputs**.

Everything is exposed through an interactive **Streamlit dashboard** with model
metrics, SHAP explanations, dataset explorer and a persistent scenario history.

## 2. Problem Statement

Can a machine-learning classifier learn to predict the outcome of an abstract
simulated conflict from only the scenario's input attributes — without seeing the
internal scoring rules? The engine's output is stochastic (adds Gaussian noise), so
the task is to model the *learnable signal* while accepting irreducible noise.

## 3. Objectives

- Build a configurable, stochastic simulation engine.
- Generate a large synthetic dataset with good coverage of the input space.
- Preprocess features and train a Random Forest classifier.
- Evaluate with real sklearn metrics (accuracy, macro precision/recall/F1, ROC-AUC).
- Explain predictions with SHAP.
- Provide a professional Streamlit dashboard with no fabricated numbers.

## 4. System Architecture

```
                          ┌───────────────────────────────┐
                          │        streamlit app.py       │
                          │  (8 pages, dark tech theme)   │
                          └────┬───────┬────────┬─────────┘
                               │       │        │
              ┌────────────────┘       │        └─────────────────┐
              v                        v                          v
   ┌──────────────┐         ┌─────────────────┐         ┌────────────────────┐
   │  Simulation  │         │  Model + SHAP   │         │  Scenario history  │
   │  engine      │         │  explainability │         │  (SQLite)          │
   └──────┬───────┘         └───────┬─────────┘         └─────────┬──────────┘
          │                         │                             │
          v                         v                             v
   ┌──────────────┐         ┌─────────────────┐         ┌────────────────────┐
   │ dataset_gen  │         │ preprocessing & │         │ src/database.py    │
   │ (LHS)        │ ──────► │ train_model     │         └────────────────────┘
   └──────┬───────┘         └─────────────────┘
          │
          v
   data/barbarik_dataset.csv
```

## 5. Dataset Generation Methodology

`src/dataset_generator.py::generate_dataset()` produces `data/barbarik_dataset.csv`
(20,000 rows, 20 columns).

- **Latin Hypercube-like sampling** for every continuous numeric attribute (force,
  mobility, logistics, intelligence, communication, readiness, difficulty,
  duration), so the full range of each feature is stratified and covered. Because
  we trained on **independent column permutations**, Side A and Side B attributes
  are decorrelated per row, so decisive outcomes remain possible.
- **Uniform random sampling** for the categorical features
  (`terrain_type`, `weather`, `visibility`, `time_of_day`).
- Each scenario gets a unique `SCN-XXXX` id, is passed through
  `simulate_scenario()`, and the resulting `outcome` is stored.

Resulting class distribution (from the actual run):

| Outcome            | Count  | Percent |
|--------------------|--------|---------|
| STALEMATE          | 11,397 | 57.0%   |
| SIDE_B_ADVANTAGE   |  4,340 | 21.7%   |
| SIDE_A_ADVANTAGE   |  4,263 | 21.3%   |

## 6. Simulation Methodology

`src/simulation_engine.py::simulate_scenario(config)`:

- **Weighted composite score** per side (no trivial "higher force wins"):
  force, mobility, logistics, intelligence, communication, readiness, each
  modulated by terrain/weather/visibility/time modifiers.
- **Terrain suitability** — e.g. `Open` rewards mobility, `Hilly` rewards a
  defender's readiness, `Urban` rewards logistics.
- **Weather** — `Extreme`/`Fog` penalise low-intelligence sides.
- **Visibility** — low visibility increases the importance of intelligence.
- **Time of day** — `Night` favours the higher-intelligence side.
- **≥2 nonlinear interaction terms**, e.g. `mobility × terrain_mobility_factor`
  and `intelligence × (1 − visibility_factor)`.
- **Gaussian noise** (σ ≈ 0.05) per side makes outcomes stochastic.
- **Outcome rule**: `SIDE_A_ADVANTAGE` if `score_a − score_b > 0.15`,
  `SIDE_B_ADVANTAGE` if the reverse, else `STALEMATE`.

## 7. Random Forest Algorithm Explanation

A **Random Forest** is an ensemble of decision trees trained on bootstrap samples
of the data, using a random subset of features at each split. Predictions are
averaged (majority vote) across trees, which reduces variance and overfitting.

Used hyperparameters (`src/model.py`):

| Parameter           | Value    | Effect                              |
|---------------------|----------|-------------------------------------|
| `n_estimators`      | 300      | More trees → smoother predictions   |
| `max_depth`         | None     | Allow deep, flexible trees          |
| `min_samples_split` | 2        | Default leaf-splitting behaviour    |
| `min_samples_leaf`  | 1        | Default, full growth                |
| `class_weight`      | "balanced" | Compensate for class imbalance    |
| `random_state`      | 42       | Reproducibility                     |
| `n_jobs`            | -1       | Use all CPU cores                   |

## 8. Feature Engineering

`src/preprocessing.py::build_preprocessing_pipeline()`:

- **Numeric features** → `StandardScaler` (good for distance- and
  gradient-based learners; harmless for tree models).
- **Categorical features** → `OneHotEncoder(handle_unknown="ignore")` so unseen
  levels at scoring time do not crash.
- Fit on the **training split only** (no leakage into the test set).
- The split is **stratified** on `outcome` (80/20, `random_state=42`).

## 9. Model Evaluation Results (from the actual run)

Real metrics computed via `sklearn.metrics` on the 4,000-row held-out test set:

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.7897 |
| Macro Precision    | 0.7767 |
| Macro Recall       | 0.7733 |
| Macro F1           | 0.7750 |
| ROC-AUC (macro, OvR)| 0.9153 |

Confusion matrix (rows = actual, cols = predicted):

```
[[ 641,    0,  212],   # SIDE_A_ADVANTAGE
 [   0,  650,  218],   # SIDE_B_ADVANTAGE
 [ 201,  210, 1868]]   # STALEMATE
```

Top features by `feature_importances_`: force strength (A & B) dominate, followed
by intelligence and communication — physically sensible for the scoring rules.
Accuracy sits in the healthy band (0.65–0.97): high enough to indicate real
learnable structure, but not so high as to imply the simulator is trivially
deterministic.

## 10. SHAP Explainability

`src/explainability.py` wraps SHAP (`shap.TreeExplainer`) for the Random Forest:

- **Global**: a summary (beeswarm) plot over sampled test rows.
- **Local**: per-instance SHAP values for the predicted class; top-8 features are
  surfaced with signed values.
- **Multi-class handling**: SHAP returns a list of per-class matrices; BARBARIK
  always selects the predicted class.
- **Fallback**: if SHAP is unavailable, it falls back to
  `model.feature_importances_` with the message
  `"SHAP unavailable, using feature_importances_ fallback."`

## 11. Dashboard Overview

`streamlit run app.py` provides 8 pages (dark military-tech analytics theme):

1. **Dashboard** — Scenario Input (sliders), abstract Simulated Battlefield,
   Predicted Outcome (donut of `predict_proba()`), SHAP importance,
   live model metrics, recent scenario history.
2. **New Scenario** — full-screen sliders to define a scenario, then simulate &
   predict it.
3. **Simulation** — animated step-by-step progress
   (INITIALIZATION → ENVIRONMENT ANALYSIS → ROUND 1–3 → OUTCOME → ML PREDICTION).
4. **Dataset** — shape, class distribution, missing values, correlation matrix,
   feature histogram, CSV download.
5. **Model Performance** — RF config, train/test sizes, metrics, confusion
   matrix heatmap, ROC curves, feature importance, classification report.
6. **Explainable AI** — SHAP summary plot, local waterfall, positive/negative
   contributions.
7. **Scenario History** — filterable table with delete and CSV export.
8. **About** — description, ethical scope, tech stack, limitations.

**Data integrity rules enforced**: no hard-coded metrics or probabilities; every
number displayed comes from a real computation (`evaluate_model`,
`predict_proba`, `simulate_scenario`, SHAP). Model + pipeline are cached with
`@st.cache_resource`; dataset/metrics with `@st.cache_data`.

## 12. Installation

```bash
# Python 3.10+
python -m venv .venv
.venv\Scripts\activate            # Windows (POSIX: source .venv/bin/activate)
pip install -r requirements.txt
```

## 13. Running the Application

```bash
# 1. (Optional) regenerate the dataset + train the model:
cd src && python -c "from dataset_generator import generate_dataset; generate_dataset(20000, '../data/barbarik_dataset.csv')"
cd .. && python train_model.py     # trains + saves models/*.joblib

# 2. Launch the dashboard:
streamlit run app.py
```

Then open http://localhost:8501.

## 14. Folder Structure

```
BARBARIK/
├── app.py                      # Streamlit entry point (nav routing)
├── train_model.py              # end-to-end train + evaluate script
├── explain.py                  # SHAP demo script
├── requirements.txt
├── README.md
├── assets/
│   └── style.css               # dark military-tech theme
├── data/                       # dataset + SQLite history
├── models/                     # random_forest_model.joblib + preprocessing_pipeline.joblib
├── assets/images/              # rendered charts (e.g. shap_summary.png)
├── pages/
│   ├── dashboard.py, scenario.py, simulation.py, dataset.py,
│   ├── model_performance.py, explainable_ai.py, history.py, about.py
├── src/
│   ├── simulation_engine.py    # simulate_scenario()
│   ├── dataset_generator.py    # generate_dataset()
│   ├── preprocessing.py        # pipeline, load, split
│   ├── model.py                # train / evaluate / save / load / predict
│   ├── explainability.py       # SHAP wrappers + fallback
│   ├── database.py             # SQLite CRUD
│   ├── ui_helpers.py           # cached model/data + plot builders
│   └── session.py              # session-state helpers
└── tests/
    ├── test_simulation.py
    ├── test_model.py
    └── test_prediction.py
```

## 15. Limitations

- The scoring weights in the engine are **hand-designed**; outcomes reflect our
  choices, not real doctrine.
- Stochastic noise (σ ≈ 0.05) bounds achievable accuracy to ~0.79 — irreducible
  uncertainty is by design.
- SHAP on a 300-tree forest is comparatively slow inside the dashboard.
- The dataset is **synthetic**; performance does not transfer to real-world data.
- Single-machine demo; not production-hardened or security-hardened.

## 16. Ethical Scope

- **Fiction only** — no real military data, targets, borders or conflicts.
- **Educational intent** — demonstrates the full ML lifecycle.
- **No policy use** — not to be used for decisions about real operations.

## 17. Future Enhancements

- Hyperparameter optimisation (cross-validated grid/random search).
- Additional learners (gradient boosting, logistic baseline) for comparison.
- Calibration curves and threshold tuning.
- A richer simulation engine (phased combat, resource attrition over rounds).
- Cross-validation-based feature selection before SHAP.
- Deployment (Docker / Streamlit Cloud).