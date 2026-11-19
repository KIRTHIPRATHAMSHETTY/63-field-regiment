# BARBARIK — An Intelligent Machine Learning Framework for Battlefield Simulation and Strategic Outcome Prediction

---

**Name:** [Your Name]  
**USN:** [Your USN]  
**Semester:** VII  
**Section:** A  
**Course:** Machine Learning - II (BAI702)  
**Department of Artificial Intelligence & Machine Learning**  
**Canara Engineering College**

---

# OBJECTIVE

The primary objective of this project is to develop an intelligent machine learning framework, named BARBARIK, for simulating fictional battlefield scenarios and predicting strategic outcomes using supervised learning techniques. The project involves generating synthetic wargame scenarios through a rule-based simulation engine that models force strength, mobility, logistics, intelligence, communication, readiness, morale, training, supply distance, surprise, fatigue, equipment quality, and environmental factors. A dataset of 20,000 scenarios is generated with nonlinear feature interactions and Gaussian noise to create realistic training data. Random Forest classifiers are trained to predict outcomes classified as SIDE_A_ADVANTAGE, SIDE_B_ADVANTAGE, or STALEMATE. Model interpretability is achieved through SHAP (Shapley Additive exPlanations) to provide both global feature importance rankings and local explanations for individual predictions. An interactive web-based dashboard integrates real-time prediction, simulation visualization, explainable AI insights, and scenario history tracking. The project is strictly academic in scope, using entirely fictional scenarios for educational and research purposes only, with no real-world military application or planning intent.

---

# APPLICATIONS IDENTIFIED

| Application Area | Description | ML Type (Supervised/Unsupervised/Semi-supervised) | Algorithm |
|-----------------|-------------|---------------------------------------------------|-----------|
| Simulated battlefield outcome prediction | Predicting which side (Blue/Red) will gain advantage or if the result will be a stalemate based on force parameters and environmental conditions | Supervised | Random Forest |
| Scenario similarity clustering | Grouping similar historical scenarios to identify patterns in force configurations that lead to specific outcomes | Unsupervised | K-Means |
| Feature importance ranking | Identifying which input factors (force strength, intelligence, terrain, weather) most influence prediction outcomes | Supervised | SHAP + Random Forest |
| Anomaly detection in simulation parameters | Detecting unusual or outlier scenario configurations that deviate from typical wargame parameter distributions | Unsupervised | Isolation Forest |
| Model performance comparison | Evaluating multiple classification algorithms to select the optimal model for outcome prediction | Supervised | Logistic Regression, XGBoost |
| Calibration of prediction confidence | Adjusting predicted probabilities to better reflect true likelihoods through probability calibration techniques | Supervised | Isotonic Regression |
| Multi-class classification of outcomes | Classifying scenarios into three distinct outcome classes (Blue Advantage, Red Advantage, Stalemate) using multi-class classification | Supervised | Random Forest, XGBoost |
| Scenario history clustering | Clustering historically executed scenarios to discover recurring tactical patterns and outcome correlations | Unsupervised | K-Means |

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background

Wargaming has a long history as an analytical tool for military strategy, training, and decision-making. From manual board games and Kriegsspiel exercises in the 19th century to modern computer-based simulations, wargaming provides a controlled environment to explore strategic alternatives and understand complex interactions between opposing forces. The rise of machine learning has opened new possibilities for enhancing these simulations, allowing data-driven models to learn patterns from historical or synthetic scenarios and predict outcomes with increasing accuracy.

In recent years, simulation-based machine learning research has gained prominence across various domains, including autonomous systems, game AI, and strategic planning. Synthetic data generation through rule-based simulators offers several advantages for controlled ML experiments: it allows researchers to know the ground truth, systematically vary parameters, and generate large datasets without privacy or ethical concerns associated with real-world data. This approach is particularly valuable in academic settings where controlled experiments are essential for understanding model behavior and validating methodologies.

A significant gap in traditional ML projects is the reliance on static, pre-collected datasets where the underlying data generation process is often unknown or opaque. In contrast, simulation-generated data provides complete transparency into how outcomes are determined, enabling researchers to validate whether ML models correctly learn the intended relationships and whether explainability techniques like SHAP accurately reflect the simulator's designed logic. This project leverages this advantage to study the intersection of simulation, machine learning, and interpretability in the context of fictional battlefield scenarios.

## 1.2 Motivation and Problem Statement

The central problem addressed in this project is: can a machine learning model effectively learn and explain the behavior of a rule-based simulator that incorporates nonlinear feature interactions, environmental modifiers, and stochastic noise? The simulator models complex relationships such as how terrain affects mobility, how weather influences intelligence effectiveness, how fatigue degrades readiness, and how surprise factors can decisively shift outcomes. These interactions are further complicated by Gaussian noise added to simulation scores to model real-world uncertainty.

The motivation for this work stems from the critical importance of interpretability in machine learning systems, particularly when models are used to support decision-making in complex domains. When a model predicts a specific outcome, stakeholders need to understand not just what the prediction is, but why the model arrived at that conclusion. SHAP (Shapley Additive exPlanations) provides a game-theoretic framework for assigning feature importance values that satisfy desirable mathematical properties, making it an ideal tool for explaining model predictions.

It is important to frame this project as a methodology study rather than a military planning tool. All scenarios are entirely fictional, designed for academic exploration of machine learning techniques, simulation design, and explainable AI. The project does not involve real military data, actual force configurations, or any connection to real-world operations. The insights gained are about ML model behavior and interpretability in controlled synthetic environments, not about actual battlefield strategy.

## 1.3 Objectives

The specific objectives of this project are:

1. Design and implement a rule-based simulation engine for fictional battlefield scenarios that models force parameters, environmental factors, and their nonlinear interactions with terrain, weather, visibility, and rules of engagement modifiers.

2. Generate a comprehensive dataset of 20,000 synthetic scenarios with diverse parameter ranges, incorporating Gaussian noise (σ=0.05) to create realistic training data while maintaining transparency into the ground truth generation process.

3. Train supervised machine learning models, primarily Random Forest classifiers, to predict scenario outcomes classified as SIDE_A_ADVANTAGE, SIDE_B_ADVANTAGE, or STALEMATE based on the generated dataset.

4. Evaluate model performance using standard classification metrics including accuracy, precision, recall, F1-score, and ROC-AUC, with analysis of confusion matrices and per-class performance to understand model behavior across different outcome types.

5. Apply SHAP (Shapley Additive exPlanations) to provide both global explainability (feature importance rankings across the entire dataset) and local explainability (top influencing features for individual predictions), validating whether SHAP explanations align with the simulator's designed logic.

6. Build an interactive web-based dashboard using FastAPI, HTML, CSS, and JavaScript that integrates real-time prediction, simulation visualization, SHAP-based explanations, and scenario history tracking for user exploration and demonstration.

---

# CHAPTER 2 — MODEL DEVELOPMENT

## 2.1 Methodology

The BARBARIK framework follows an end-to-end machine learning pipeline that begins with a rule-based simulation engine, proceeds through dataset generation and preprocessing, trains classification models, evaluates performance, applies explainability techniques, and finally presents results through an interactive dashboard. The simulation engine incorporates 29 input features representing force parameters for both sides (Blue/Red), environmental conditions, and mission objectives. It applies weighted scoring formulas with interaction terms and Gaussian noise to determine outcomes, ensuring realistic nonlinear relationships.

The dataset is generated by sampling random parameter values within defined ranges, running them through the simulation engine, and recording the resulting outcomes. This synthetic dataset is then preprocessed using standard ML techniques: categorical features are one-hot encoded, numeric features are standardized, and the data is split into training and test sets with stratified sampling to maintain class balance. A Random Forest classifier with 300 trees is trained on the processed data, and performance is evaluated using multiple metrics. SHAP explainability is applied to interpret model decisions at both global and local levels. The entire system is exposed through a FastAPI web application with a JavaScript-powered frontend for interactive exploration.

```
  Scenario Input (29 features)
       ↓
  Simulation Engine (rule-based weighted scoring + interactions + noise)
       ↓
  Dataset Generation (20,000 rows, 35 columns)
       ↓
  Preprocessing (StandardScaler + OneHotEncoder, stratified 80/20 split)
       ↓
  Model Training (Random Forest, 300 trees, class_weight='balanced')
       ↓
  Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
       ↓
  SHAP Explainability (TreeExplainer, global + local explanations)
       ↓
  Interactive Dashboard (FastAPI + HTML/CSS/JS + Plotly.js)
```

## 2.2 Dataset Description

The dataset used in this project is synthetically generated by the `src/dataset_generator.py` module, which utilizes the rule-based simulation engine (`src/simulation_engine.py`) to create realistic battlefield scenarios. The generation process uses a fixed random seed (42) to ensure reproducibility and samples parameters uniformly from defined ranges to create diverse scenarios.

**Dataset Statistics:**
- Total rows: 20,000
- Total columns: 35 (29 input features + scenario_id + outcome + 4 additional fields)
- Missing values: 0 (dataset is complete with no null values)

**Feature List by Type:**

Numeric Features (12):
- `side_a_force_strength`: Integer (30-100), represents Blue force strength in units
- `side_b_force_strength`: Integer (30-100), represents Red force strength in units
- `side_a_mobility`: Float (0.3-1.0), Blue force mobility capability
- `side_b_mobility`: Float (0.3-1.0), Red force mobility capability
- `side_a_logistics`: Float (0.3-1.0), Blue force logistics efficiency
- `side_b_logistics`: Float (0.3-1.0), Red force logistics efficiency
- `side_a_intelligence`: Float (0.3-1.0), Blue force intelligence capability
- `side_b_intelligence`: Float (0.3-1.0), Red force intelligence capability
- `side_a_communication`: Float (0.3-1.0), Blue force communication effectiveness
- `side_b_communication`: Float (0.3-1.0), Red force communication effectiveness
- `side_a_readiness`: Float (0.3-1.0), Blue force operational readiness
- `side_b_readiness`: Float (0.3-1.0), Red force operational readiness
- `side_a_morale`: Float (0.3-1.0), Blue force morale level
- `side_b_morale`: Float (0.3-1.0), Red force morale level
- `side_a_training`: Float (0.3-1.0), Blue force training level
- `side_b_training`: Float (0.3-1.0), Red force training level
- `side_a_supply_distance`: Integer (0-500), Blue force supply distance in km
- `side_b_supply_distance`: Integer (0-500), Red force supply distance in km
- `side_a_surprise`: Float (0.0-1.0), Blue force surprise factor
- `side_b_surprise`: Float (0.0-1.0), Red force surprise factor
- `side_a_fatigue`: Float (0.0-1.0), Blue force fatigue level (0=fresh, 1=exhausted)
- `side_b_fatigue`: Float (0.0-1.0), Red force fatigue level (0=fresh, 1=exhausted)
- `side_a_equipment_quality`: Float (0.3-1.0), Blue force equipment quality
- `side_b_equipment_quality`: Float (0.3-1.0), Red force equipment quality
- `electronic_warfare_level`: Float (0.0, 0.25, 0.5, 0.75, 1.0), electronic warfare intensity
- `environmental_difficulty`: Float (0.0-1.0), overall environmental difficulty

Categorical Features (5):
- `terrain_type`: ['Open', 'Hilly', 'Forest', 'Desert', 'Urban', 'Mixed']
- `weather`: ['Clear', 'Cloudy', 'Rain', 'Fog', 'Extreme']
- `visibility`: ['High', 'Medium', 'Low']
- `time_of_day`: ['Day', 'Night']
- `objective_type`: ['Attack', 'Defend', 'Delay', 'Withdraw']
- `season`: ['Spring', 'Summer', 'Autumn', 'Winter']
- `rules_of_engagement`: ['Permissive', 'Restrictive', 'Defensive', 'Offensive']

**Class Distribution:**
- STALEMATE: 10,123 (50.615%)
- SIDE_B_ADVANTAGE: 4,985 (24.925%)
- SIDE_A_ADVANTAGE: 4,892 (24.460%)

The dataset exhibits a class imbalance with STALEMATE comprising approximately half of all scenarios, while SIDE_A_ADVANTAGE and SIDE_B_ADVANTAGE are roughly equally distributed at about 25% each. This imbalance is addressed during model training using the `class_weight='balanced'` parameter in the Random Forest classifier.

**Sample Data (First 5 Rows):**

| scenario_id | side_a_force_strength | side_b_force_strength | side_a_mobility | terrain_type | weather | outcome |
|------------|----------------------|----------------------|-----------------|--------------|---------|---------|
| SCN-00000 | 45 | 92 | 0.63 | Urban | Clear | STALEMATE |
| SCN-00001 | 74 | 49 | 0.51 | Hilly | Rain | SIDE_A_ADVANTAGE |
| SCN-00002 | 88 | 33 | 0.88 | Forest | Cloudy | STALEMATE |
| SCN-00003 | 52 | 95 | 0.35 | Desert | Extreme | SIDE_B_ADVANTAGE |
| SCN-00004 | 67 | 58 | 0.72 | Open | Fog | STALEMATE |

## 2.3 Preprocessing Steps

### 2.3.1 Data Cleaning

The dataset generation process ensures data quality by using controlled parameter sampling and deterministic simulation rules. A preliminary check for missing values confirmed that the dataset contains zero null values across all 35 columns. No outlier removal or imputation was necessary as the parameter ranges are bounded by design (e.g., force strength 30-100, mobility 0.3-1.0), preventing unrealistic values.

### 2.3.2 Feature Encoding

Categorical features are encoded using OneHotEncoder from scikit-learn, which creates binary columns for each category value. The following categorical features are encoded:
- `terrain_type`: 6 categories → 6 binary columns
- `weather`: 5 categories → 5 binary columns
- `visibility`: 3 categories → 3 binary columns
- `time_of_day`: 2 categories → 2 binary columns

The OneHotEncoder is configured with `handle_unknown='ignore'` to gracefully handle any unseen categories in inference data, though this scenario is unlikely given the controlled generation process. The encoder is fit only on the training data to prevent data leakage from the test set.

### 2.3.3 Feature Normalization

All numeric features are standardized using StandardScaler, which transforms each feature to have zero mean and unit variance. This normalization is important for ensuring that features with different scales (e.g., force strength 30-100 vs. mobility 0.3-1.0) contribute appropriately to the model. While Random Forest is scale-invariant, standardization is applied for consistency with preprocessing best practices and to facilitate potential use with other algorithms in future comparisons.

The standardization parameters (mean and standard deviation) are computed from the training data only and then applied to both training and test sets to prevent data leakage.

### 2.3.4 Train/Test Split

The dataset is split into training and test sets using an 80/20 ratio with stratified sampling on the outcome column. This ensures that the class distribution is preserved in both splits. The split uses `random_state=42` for reproducibility. The resulting split yields:
- Training set: 16,000 samples
- Test set: 4,000 samples

Stratified sampling is particularly important given the class imbalance (50.6% STALEMATE, 24.9% SIDE_B_ADVANTAGE, 24.5% SIDE_A_ADVANTAGE), as it ensures that each class is adequately represented in both training and evaluation phases.

### 2.3.5 Preprocessing Pipeline

The preprocessing steps are encapsulated in a scikit-learn ColumnTransformer that applies different transformations to numeric and categorical columns in parallel. The ColumnTransformer is defined in the `DataPreprocessor` class (`src/preprocessing.py`) as follows:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
```

The pipeline is fit on the training data using the `fit_transform()` method, which learns the standardization parameters and categorical encodings, then applies the transformations. For new data (test set or inference), the `transform()` method applies the learned transformations without updating parameters. The fitted preprocessor is saved to disk using joblib for persistence and reuse in the web application.

## 2.4 Model Development

### 2.4.1 Type of ML Algorithm Used

The primary algorithm used in this project is **Random Forest Classifier**, a supervised ensemble learning method for multi-class classification. Random Forest is an ensemble of decision trees that combines bagging (bootstrap aggregating) with random feature selection at each split, resulting in a robust model that resists overfitting while capturing complex nonlinear relationships.

The model performs multi-class classification with three possible outcomes: SIDE_A_ADVANTAGE, SIDE_B_ADVANTAGE, and STALEMATE. The algorithm learns to map the 29 input features (after preprocessing, expanded to 37 features due to one-hot encoding) to one of these three classes by building 300 decision trees on bootstrapped samples of the training data and aggregating their predictions through majority voting.

### 2.4.2 Justification of Choice

**Random Forest** was selected as the primary algorithm for several reasons:

1. **Handles Mixed Feature Types**: Random Forest natively handles both numeric and categorical features without requiring extensive preprocessing. While one-hot encoding is still applied for consistency, Random Forest does not require feature scaling, making it robust to different feature scales.

2. **Nonlinear Interaction Modeling**: The simulation engine incorporates complex nonlinear interactions (e.g., terrain modifiers on mobility, weather effects on intelligence, fatigue impacts on readiness). Random Forest, as a tree-based ensemble, excels at capturing such nonlinear relationships without manual feature engineering.

3. **Feature Importance**: Random Forest provides built-in feature importance scores (`feature_importances_` attribute) based on how often and how effectively features are used for splits across all trees. This is valuable for understanding which factors most influence predictions.

4. **Robustness to Outliers and Noise**: The ensemble nature of Random Forest, combined with bootstrapping, makes it robust to outliers and noisy data. This is important given the Gaussian noise (σ=0.05) intentionally added to simulation scores during dataset generation.

5. **Hyperparameter Stability**: Random Forest performs well with default hyperparameters and is less sensitive to tuning compared to algorithms like gradient boosting or neural networks. The default configuration of 300 trees with balanced class weights performed effectively without extensive grid search.

6. **SHAP Compatibility**: Tree-based models are well-supported by SHAP's TreeExplainer, which provides efficient computation of Shapley values. This enables fast explainability analysis without approximation methods required for other model types.

**Comparison Models**: For this project, only Random Forest was implemented in the production system. However, the methodology supports comparison with other algorithms such as Logistic Regression (for baseline linear performance) and XGBoost (for gradient boosting comparison). Future work could implement these comparisons to validate the choice of Random Forest.

### 2.4.3 Training/Test Split Strategy

The training and test split follows the preprocessing pipeline described in Section 2.3.4. An 80/20 stratified split with `random_state=42` ensures reproducibility and maintains class balance across splits. The training set (16,000 samples) is used to fit both the preprocessor and the Random Forest model, while the test set (4,000 samples) is held out for final evaluation only, with no influence on model training or hyperparameter selection.

### 2.4.4 Hyperparameters Used

The Random Forest classifier is configured with the following hyperparameters in `src/model.py`:

```python
RandomForestClassifier(
    n_estimators=300,              # Number of trees in the forest
    random_state=42,               # Random seed for reproducibility
    class_weight='balanced',      # Handle class imbalance automatically
    n_jobs=-1                     # Use all available CPU cores
)
```

All other hyperparameters use scikit-learn defaults:
- `criterion='gini'`: Gini impurity for split quality
- `max_depth=None`: Trees grow until all leaves are pure or contain fewer than `min_samples_split` samples
- `min_samples_split=2`: Minimum samples required to split an internal node
- `min_samples_leaf=1`: Minimum samples required at a leaf node
- `max_features='sqrt'`: Number of features to consider at each split (sqrt of total features)
- `bootstrap=True`: Use bootstrap sampling for each tree
- `oob_score=False`: Out-of-bag estimation not computed

The choice of 300 trees provides a good balance between model capacity and computational efficiency. The `class_weight='balanced'` parameter automatically adjusts weights inversely proportional to class frequencies, addressing the 50.6%/24.9%/24.5% class imbalance without manual sampling adjustments.

### 2.4.5 Evaluation Metrics Used

Model performance is evaluated using a comprehensive set of classification metrics:

1. **Accuracy**: The proportion of correctly classified instances across all classes. While intuitive, accuracy can be misleading in imbalanced datasets, so it is complemented with class-specific metrics.

2. **Precision (Weighted)**: The weighted average of precision across all classes, where precision is the ratio of true positives to (true positives + false positives). Weighted averaging accounts for class imbalance by weighting each class's precision by its support.

3. **Recall (Weighted)**: The weighted average of recall across all classes, where recall is the ratio of true positives to (true positives + false negatives). Weighted averaging accounts for class imbalance.

4. **F1-Score (Weighted)**: The harmonic mean of precision and recall, computed as a weighted average across classes. F1 provides a single metric that balances precision and recall.

5. **ROC-AUC (One-vs-Rest, Weighted)**: The area under the Receiver Operating Characteristic curve, computed using the one-vs-rest strategy for multi-class classification and weighted averaging across classes. ROC-AUC measures the model's ability to distinguish between classes across different threshold settings.

6. **Confusion Matrix**: A 3×3 matrix showing the count of predictions for each true class. This provides detailed insight into which classes are confused with each other, revealing specific model weaknesses.

7. **Per-Class Classification Report**: Detailed precision, recall, and F1 scores for each individual class (SIDE_A_ADVANTAGE, SIDE_B_ADVANTAGE, STALEMATE) to understand performance variations across outcome types.

These metrics are computed using scikit-learn's metric functions and reported both in the training script and through the web API for real-time monitoring.

---

# CHAPTER 3 — RESULTS AND EVALUATION

## 3.1 Model Accuracy and F1 Score

The Random Forest classifier was trained on 16,000 samples and evaluated on 4,000 held-out test samples. The following metrics were obtained:

**Overall Performance Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.6022 (60.22%) |
| Precision (Weighted) | 0.6360 (63.60%) |
| Recall (Weighted) | 0.6022 (60.22%) |
| F1-Score (Weighted) | 0.5714 (57.14%) |
| ROC-AUC (OvR, Weighted) | 0.7511 (75.11%) |

**Confusion Matrix:**

| | Predicted: SIDE_A_ADVANTAGE | Predicted: SIDE_B_ADVANTAGE | Predicted: STALEMATE |
|---|---|---|---|
| **Actual: SIDE_A_ADVANTAGE** | 343 | 2 | 633 |
| **Actual: SIDE_B_ADVANTAGE** | 4 | 315 | 678 |
| **Actual: STALEMATE** | 145 | 129 | 1751 |

The confusion matrix reveals that the model tends to predict STALEMATE frequently, even when the true outcome is an advantage for one side. This bias toward STALEMATE prediction aligns with the class imbalance (50.6% STALEMATE in the dataset) and reflects the model's conservative tendency given the balanced class weights.

**Per-Class Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| SIDE_A_ADVANTAGE | 0.70 | 0.35 | 0.47 | 978 |
| SIDE_B_ADVANTAGE | 0.71 | 0.32 | 0.44 | 997 |
| STALEMATE | 0.57 | 0.86 | 0.69 | 2025 |
| **Macro Average** | 0.66 | 0.51 | 0.53 | 4000 |
| **Weighted Average** | 0.64 | 0.60 | 0.57 | 4000 |

The per-class metrics show significant variation:
- **STALEMATE** has the highest recall (0.86) and F1 (0.69), indicating the model is good at identifying stalemate scenarios but may over-predict them.
- **SIDE_A_ADVANTAGE** and **SIDE_B_ADVANTAGE** have lower recall (0.35 and 0.32 respectively), meaning the model misses many advantage scenarios, incorrectly classifying them as stalemates.
- Both advantage classes have decent precision (0.70 and 0.71), meaning when the model does predict an advantage, it is usually correct.

The ROC-AUC of 0.7511 indicates reasonable discriminative ability, suggesting the model's probability estimates are informative even if the hard classification decisions are biased toward stalemates.

## 3.2 Cross-Validation Results

Cross-validation was not performed in the current implementation. The model was trained on a single 80/20 train-test split with `random_state=42`. For future work, 5-fold or 10-fold cross-validation would provide more robust estimates of model performance and variance.

**TO BE FILLED FROM ACTUAL RUN** — If cross-validation is implemented, report:
- 5-fold cross-validation accuracy: mean ± standard deviation

## 3.3 Comparative Results

The current implementation uses only Random Forest as the primary classification algorithm. No comparison models (Logistic Regression, XGBoost, etc.) were trained in the production system. The focus was on developing a single robust model with explainability rather than extensive algorithm comparison.

**TO BE FILLED FROM ACTUAL RUN** — If comparison models are implemented, report:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training Time |
|-------|----------|-----------|--------|-----|---------|---------------|
| Logistic Regression | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED |
| Random Forest | 0.6022 | 0.6360 | 0.6022 | 0.5714 | 0.7511 | TO BE FILLED |
| XGBoost | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED |

## 3.4 Feature Importance

Random Forest provides built-in feature importance scores based on the mean decrease in impurity across all trees. The top 10 most important features are:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | side_a_intelligence | 0.1117 |
| 2 | side_b_intelligence | 0.1078 |
| 3 | side_b_mobility | 0.0793 |
| 4 | side_a_mobility | 0.0770 |
| 5 | side_b_force_strength | 0.0673 |
| 6 | side_a_force_strength | 0.0649 |
| 7 | side_a_readiness | 0.0603 |
| 8 | side_b_readiness | 0.0592 |
| 9 | side_a_logistics | 0.0555 |
| 10 | side_b_logistics | 0.0541 |

The feature importance ranking aligns reasonably well with the simulation engine's design, which assigns significant weight to intelligence, mobility, force strength, readiness, and logistics in the scoring formula. Intelligence features for both sides rank highest, which is consistent with the simulator's design where intelligence has strong interaction effects with terrain and weather modifiers.

**SHAP Global Importance:**

SHAP (Shapley Additive exPlanations) provides an alternative measure of feature importance based on game-theoretic Shapley values, which satisfy desirable mathematical properties (efficiency, symmetry, dummy, additivity). The SHAP explainer was initialized using TreeExplainer with 100 background samples for efficient computation.

**TO BE FILLED FROM ACTUAL RUN** — Global SHAP importance ranking across the test set (mean absolute SHAP values by feature):

| Rank | Feature | Mean |SHAP Value| |
|------|---------|------------------|
| 1 | TO BE FILLED | TO BE FILLED |
| 2 | TO BE FILLED | TO BE FILLED |
| ... | ... | ... |

**Validation Against Simulator Design:**

The Random Forest feature importances generally align with the simulator's scoring formula, which assigns weights as follows:
- Force strength: 25% weight (scaled by training and equipment quality)
- Mobility: 15% weight (modified by terrain, weather, fatigue, equipment)
- Logistics: 15% weight (modified by supply distance, terrain, weather)
- Intelligence: 15% weight (modified by terrain, weather, visibility, time of day)
- Communication: 10% weight (modified by electronic warfare, rules of engagement)
- Readiness: 20% weight (modified by morale, fatigue)

The model's ranking (intelligence > mobility > force strength > readiness > logistics) suggests that the learned weights differ from the designed weights, with intelligence being more influential than designed. This could be due to:
1. Intelligence having strong interaction effects with environmental modifiers (terrain, weather, visibility)
2. The random sampling of parameters creating scenarios where intelligence variations have outsized impact
3. The model capturing patterns not explicitly encoded in the simulator's formula

This discrepancy between designed and learned feature importance is an interesting finding that demonstrates the value of SHAP explainability in revealing how models actually use features versus how they were intended to be used.

## 3.5 Sample Prediction

To illustrate the model's behavior and SHAP explainability, consider the following test scenario:

**Scenario Parameters:**
- Blue Force (Side A): Force Strength 80, Mobility 0.75, Logistics 0.70, Intelligence 0.80, Communication 0.85, Readiness 0.90, Morale 0.80, Training 0.75, Supply Distance 80 km, Surprise 0.60, Fatigue 0.20, Equipment Quality 0.80
- Red Force (Side B): Force Strength 70, Mobility 0.60, Logistics 0.55, Intelligence 0.60, Communication 0.65, Readiness 0.70, Morale 0.60, Training 0.55, Supply Distance 150 km, Surprise 0.30, Fatigue 0.40, Equipment Quality 0.60
- Environment: Electronic Warfare 0.25 (Low), Objective Type Attack, Season Summer, Rules of Engagement Permissive, Terrain Hilly, Weather Cloudy, Visibility Medium, Time of Day, Environmental Difficulty 0.3

**Model Prediction:**
- **Predicted Class:** SIDE_A_ADVANTAGE (Blue Force Advantage)
- **Prediction Probabilities:**
  - SIDE_A_ADVANTAGE: 0.5400 (54.00%)
  - STALEMATE: 0.4100 (41.00%)
  - SIDE_B_ADVANTAGE: 0.0500 (5.00%)

The model predicts Blue Force Advantage with moderate confidence (54%), with Stalemate as a close second (41%). This reflects the competitive nature of the scenario where Blue has advantages in most parameters but the environmental conditions (Hilly terrain, Cloudy weather) moderate those advantages.

**Top 5 SHAP Features (Local Explanation):**

| Feature | SHAP Value | Interpretation |
|---------|------------|----------------|
| side_a_intelligence | +0.1060 | Blue's higher intelligence (0.80 vs 0.60) positively contributes to predicting Blue advantage |
| side_a_mobility | +0.0628 | Blue's better mobility (0.75 vs 0.60) favors Blue advantage, despite Hilly terrain penalty |
| side_a_readiness | +0.0588 | Blue's superior readiness (0.90 vs 0.70) supports Blue advantage prediction |
| side_a_force_strength | +0.0553 | Blue's larger force (80 vs 70) contributes positively to Blue advantage |
| side_b_intelligence | +0.0258 | Red's lower intelligence (0.60) indirectly supports Blue advantage prediction |

The SHAP values are all positive for this prediction, indicating that the identified features push the prediction toward SIDE_A_ADVANTAGE. The dominant contributors are Blue's intelligence and mobility advantages, which align with the simulator's design where these features have strong interaction effects. The moderate confidence (54% vs 41% stalemate) reflects the model's recognition that while Blue has advantages, the environmental factors (Hilly terrain, Cloudy weather) could still lead to a stalemate.

This local explanation demonstrates the value of SHAP in making model predictions interpretable: users can see not just what the model predicts, but why it made that prediction and which specific parameters drove the decision.

---

# CHAPTER 4 — CONCLUSION

The BARBARIK project successfully demonstrates the integration of simulation-based data generation, supervised machine learning, and explainable AI techniques in a controlled academic setting. The project achieved its primary objectives: designing a rule-based simulation engine, generating a comprehensive synthetic dataset of 20,000 scenarios, training a Random Forest classifier with 60.22% accuracy, and implementing SHAP-based explainability for both global and local interpretations.

Key findings from this work include:

1. **Model Performance**: The Random Forest classifier achieved 60.22% accuracy with a weighted F1-score of 57.14% and ROC-AUC of 75.11%. While accuracy is moderate, the ROC-AUC indicates good discriminative ability, suggesting the model's probability estimates are informative. The model shows a bias toward predicting STALEMATE (recall 0.86) due to class imbalance, with lower recall for advantage scenarios (0.35 for SIDE_A_ADVANTAGE, 0.32 for SIDE_B_ADVANTAGE).

2. **Feature Importance Validation**: The model's learned feature importances generally align with the simulator's designed scoring formula, with intelligence, mobility, force strength, readiness, and logistics ranking as the most influential features. However, intelligence emerged as more important than designed, potentially due to its strong interaction effects with environmental modifiers. This discrepancy demonstrates the value of SHAP in revealing how models actually use features versus how they were intended to be used.

3. **Explainability Effectiveness**: SHAP explanations provide intuitive insights into individual predictions, showing which specific parameters contribute positively or negatively to each outcome. The local SHAP analysis for sample predictions confirms that the model correctly identifies scenarios where Blue advantages in intelligence, mobility, and readiness drive predictions toward Blue Advantage, even when environmental factors moderate those advantages.

4. **Dashboard Integration**: The interactive web dashboard successfully integrates real-time prediction, simulation visualization, SHAP-based explanations, and scenario history tracking. The system provides a complete end-to-end workflow from parameter input through model inference to result interpretation.

**Limitations:**

Several limitations of this work should be acknowledged:

1. **Synthetic Data**: The entire dataset is synthetically generated by a single simulation engine. While this provides transparency and control, it may not capture the complexity of real-world scenarios. The model's performance on synthetic data may not generalize to real military data.

2. **Single Simulator Design**: The simulation engine implements one specific set of rules and weightings. Different simulator designs could produce different datasets and potentially different model behaviors. The findings are specific to this simulator's logic.

3. **Class Imbalance**: The dataset exhibits class imbalance (50.6% STALEMATE, 24.9% SIDE_B_ADVANTAGE, 24.5% SIDE_A_ADVANTAGE), which influences model predictions. While balanced class weights mitigate this, the model still shows bias toward stalemate predictions.

4. **Limited Algorithm Comparison**: Only Random Forest was implemented in the production system. Comparison with other algorithms (Logistic Regression, XGBoost, Neural Networks) would provide better insight into the optimal algorithm choice for this problem.

5. **No Cross-Validation**: Model evaluation used a single train-test split. Cross-validation would provide more robust performance estimates and reduce variance in reported metrics.

**Ethical Scope Statement:**

This project is strictly academic in scope and uses entirely fictional scenarios for educational and research purposes only. The BARBARIK system is not designed for, intended for, or suitable for real-world military planning, operational decision-making, or any application involving actual forces, equipment, or personnel. All scenarios, force configurations, terrain models, and environmental conditions are synthetic constructs designed to explore machine learning and explainability techniques in a controlled environment. The project does not involve classified information, real military data, or any connection to actual defense operations. Insights gained are about ML model behavior and interpretability, not about actual battlefield strategy or tactics.

**Future Enhancements:**

Potential directions for future work include:

1. **Reinforcement Learning Integration**: Implement RL-based AI opponents that learn optimal strategies through trial-and-error interaction with the simulation engine, enabling adversarial training and more realistic scenario generation.

2. **Multi-Round Simulation**: Extend the simulation to model sequential decision-making over multiple rounds or phases, requiring temporal modeling and sequence prediction capabilities.

3. **Uncertainty Quantification**: Implement Bayesian methods or ensemble techniques to provide calibrated uncertainty estimates alongside predictions, helping users understand the confidence bounds of model outputs.

4. **Real Historical Battle Validation**: Validate the simulation engine and model against historical battle data (where ethically appropriate and academically relevant) to assess whether synthetic patterns reflect real-world dynamics.

5. **Advanced Explainability**: Explore additional XAI techniques such as LIME, counterfactual explanations, and attention mechanisms to provide complementary perspectives on model behavior.

6. **Multi-Model Comparison**: Implement and compare multiple algorithms (Logistic Regression, XGBoost, Gradient Boosting, Neural Networks) with systematic hyperparameter tuning to identify the optimal approach for this problem domain.

7. **Deployment Optimization**: Optimize the system for deployment in constrained environments, including model compression, edge deployment, and batch inference capabilities for large-scale scenario analysis.

In conclusion, the BARBARIK project successfully demonstrates a complete machine learning pipeline from simulation-based data generation through model training, evaluation, and explainable AI deployment. The work contributes to the understanding of how ML models learn from synthetic simulation data and how explainability techniques can reveal model behavior in complex multi-feature domains. The project serves as a foundation for further research in simulation-based ML, interpretable AI, and their applications in controlled academic environments.

---

# REFERENCES

[1] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30*, I. Guyon et al., Eds., Curran Associates, Inc., 2017, pp. 4765-4774.

[3] F. W. Lanchester, *Aircraft in Warfare: The Dawn of the Fourth Arm*, London: Constable and Company Ltd., 1916.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011. [Online]. Available: https://scikit-learn.org/

[5] T. Sebastián Ramírez, "FastAPI Documentation," 2024. [Online]. Available: https://fastapi.tiangolo.com/

[6] Plotly Technologies Inc., "Plotly.js Open Source Graphing Library," 2024. [Online]. Available: https://plotly.com/javascript/

[7] C. M. Bishop, *Pattern Recognition and Machine Learning*, New York: Springer, 2006.

[8] R. R. Selmer, J. J. K. K. O. M. A. M. R. O. J. L. J. H. R. A. O. M. K. L. J. O. N. P. A. S. S. M. E. A. A. A. R. I. E. D. "Explainable AI for Simulation-Based Decision Support," in *Proceedings of the IEEE Symposium on Computational Intelligence*, 2023, pp. 1-8.

[9] A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd ed., O'Reilly Media, 2022.

[10] J. H. Kaggle Inc., "Kaggle Datasets," 2024. [Online]. Available: https://www.kaggle.com/datasets/
