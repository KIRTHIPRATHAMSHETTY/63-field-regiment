import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
from typing import Dict, Any, Optional
from .preprocessing import DataPreprocessor


class OutcomePredictor:
    """XGBoost model for predicting military scenario outcomes."""
    
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric='mlogloss'
        )
        self.preprocessor = DataPreprocessor()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the XGBoost model on the given dataset.
        
        Args:
            df: DataFrame with features and outcome
        
        Returns:
            Dictionary with training metrics
        """
        # Preprocess data (fit on training only)
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)
        
        # Encode target labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.model.fit(X_train, y_train_encoded)
        self.is_trained = True
        
        # Evaluate
        y_pred_encoded = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        # Feature importances
        feature_names = self._get_feature_names()
        importances = dict(zip(feature_names, [float(v) for v in self.model.feature_importances_]))
        
        # ROC curve data
        roc_curve_data = self._get_roc_curve_data(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'roc_curve_data': roc_curve_data,
            'feature_importances': importances
        }
        
        return metrics
    
    def predict_outcome(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict outcome for a single scenario.
        
        Args:
            scenario: Dictionary with scenario parameters
        
        Returns:
            Dictionary with prediction and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to DataFrame
        df = pd.DataFrame([scenario])
        
        # Add dummy scenario_id and outcome if missing
        if 'scenario_id' not in df.columns:
            df['scenario_id'] = 'SCN-PREDICT'
        if 'outcome' not in df.columns:
            df['outcome'] = 'STALEMATE'  # Dummy, will be dropped
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Predict
        pred_encoded = self.model.predict(X)[0]
        predicted_class = self.label_encoder.inverse_transform([pred_encoded])[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Map probabilities to class names
        class_names = self.label_encoder.classes_
        prob_dict = {str(cls_): float(prob) for cls_, prob in zip(class_names, probabilities)}
        
        return {
            'predicted_class': predicted_class,
            'probabilities': prob_dict
        }
    
    def _get_feature_names(self) -> list:
        """Get feature names after preprocessing."""
        numeric_names = self.preprocessor.numeric_features
        categorical_encoder = self.preprocessor.preprocessor.named_transformers_['cat']
        categorical_names = categorical_encoder.get_feature_names_out(self.preprocessor.categorical_features)
        return list(numeric_names) + list(categorical_names)
    
    def _get_roc_curve_data(self, y_test, y_pred_proba) -> Dict[str, Any]:
        """Generate ROC curve data for visualization."""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve
        
        classes = self.label_encoder.classes_
        y_test_bin = label_binarize(y_test, classes=classes)
        
        roc_data = {}
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_data[str(class_name)] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        
        return roc_data
    
    def save(self, model_path: str, preprocessor_path: str, label_encoder_path: Optional[str] = None):
        """Save model, preprocessor, and label encoder to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        if label_encoder_path is None:
            label_encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder.joblib')
        os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        self.preprocessor.save(preprocessor_path)
        joblib.dump(self.label_encoder, label_encoder_path)
    
    def load(self, model_path: str, preprocessor_path: str, label_encoder_path: Optional[str] = None):
        """Load model, preprocessor, and label encoder from disk."""
        self.model = joblib.load(model_path)
        self.preprocessor.load(preprocessor_path)
        
        if label_encoder_path is None:
            label_encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder.joblib')
        
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            # Reconstruct default classes if missing
            self.label_encoder = LabelEncoder()
            if hasattr(self.model, 'classes_'):
                # In XGBoost with LabelEncoder, classes_ is [0, 1, 2]
                self.label_encoder.classes_ = np.array(['SIDE_A_ADVANTAGE', 'SIDE_B_ADVANTAGE', 'STALEMATE'])
            else:
                self.label_encoder.classes_ = np.array(['SIDE_A_ADVANTAGE', 'SIDE_B_ADVANTAGE', 'STALEMATE'])
        
        self.is_trained = True


if __name__ == '__main__':
    # Test model training
    from .dataset_generator import generate_dataset
    
    print("Training Random Forest model...")
    df = generate_dataset(5000)  # Smaller dataset for testing
    
    predictor = OutcomePredictor()
    metrics = predictor.train(df)
    
    print(f"\nModel Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Test prediction
    test_scenario = {
        'scenario_id': 'SCN-TEST',
        'side_a_force_strength': 80,
        'side_b_force_strength': 70,
        'side_a_mobility': 0.75,
        'side_b_mobility': 0.60,
        'side_a_logistics': 0.70,
        'side_b_logistics': 0.55,
        'side_a_intelligence': 0.80,
        'side_b_intelligence': 0.60,
        'side_a_communication': 0.85,
        'side_b_communication': 0.65,
        'side_a_readiness': 0.75,
        'side_b_readiness': 0.60,
        'terrain_type': 'Hilly',
        'weather': 'Cloudy',
        'visibility': 'Medium',
        'time_of_day': 'Day',
        'environmental_difficulty': 0.3,
        'outcome': 'STALEMATE'
    }
    
    prediction = predictor.predict_outcome(test_scenario)
    print(f"\nTest Prediction:")
    print(f"Predicted: {prediction['predicted_class']}")
    print(f"Probabilities: {prediction['probabilities']}")
