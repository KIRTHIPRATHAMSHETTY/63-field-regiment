import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple
from .model import OutcomePredictor


class ModelExplainer:
    """SHAP-based model explainability."""
    
    def __init__(self, predictor: OutcomePredictor):
        self.predictor = predictor
        self.explainer = None
        self.background_data = None
    
    def initialize_explainer(self, background_data: pd.DataFrame, n_background: int = 100):
        """
        Initialize SHAP explainer with XGBoost model.
        
        Args:
            background_data: DataFrame for background samples
            n_background: Number of background samples to use
        """
        # Preprocess background data
        background_processed = self.predictor.preprocessor.transform(
            background_data.head(n_background)
        )
        
        # Use TreeExplainer for XGBoost
        try:
            # TreeExplainer with tree_path_dependent is optimal for XGBoost
            self.explainer = shap.TreeExplainer(self.predictor.model)
            self.background_data = background_processed
        except Exception:
            try:
                self.explainer = shap.TreeExplainer(
                    self.predictor.model,
                    data=background_processed,
                    feature_perturbation="interventional"
                )
                self.background_data = background_processed
            except Exception as e:
                print(f"SHAP TreeExplainer failed: {e}")
                self.explainer = None
    
    def get_shap_values(self, scenario: Dict[str, Any]) -> np.ndarray:
        """
        Get SHAP values for a single scenario.
        
        Args:
            scenario: Dictionary with scenario parameters
        
        Returns:
            SHAP values array (1D array of floats)
        """
        feature_names = self._get_feature_names()
        if self.explainer is None:
            # Fallback: return zeros
            return np.zeros(len(feature_names))
        
        # Convert to DataFrame and preprocess
        df = pd.DataFrame([scenario])
        if 'scenario_id' not in df.columns:
            df['scenario_id'] = 'SCN-SHAP'
        if 'outcome' not in df.columns:
            df['outcome'] = 'STALEMATE'
        
        X = self.predictor.preprocessor.transform(df)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Determine class index
        pred = self.predictor.model.predict(X)[0]
        if hasattr(self.predictor, 'label_encoder') and hasattr(self.predictor.label_encoder, 'classes_'):
            classes_list = list(self.predictor.label_encoder.classes_)
        else:
            classes_list = list(getattr(self.predictor.model, 'classes_', [0, 1, 2]))
        
        if isinstance(pred, (int, np.integer)):
            class_idx = int(pred)
            if class_idx < 0 or class_idx >= len(classes_list):
                class_idx = 0
        elif pred in classes_list:
            class_idx = classes_list.index(pred)
        else:
            class_idx = 0
        
        # Extract 1D array for the target class
        if isinstance(shap_values, list):
            res = shap_values[class_idx][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Shape could be (1, n_features, n_classes) or (1, n_classes, n_features)
                if shap_values.shape[2] == len(classes_list):
                    res = shap_values[0, :, class_idx]
                else:
                    res = shap_values[0, class_idx, :]
            elif shap_values.ndim == 2:
                res = shap_values[0]
            else:
                res = shap_values.flatten()
        else:
            res = np.zeros(len(feature_names))
            
        return np.array([float(x) for x in res.flatten()])
    
    def get_top_influencing_features(self, scenario: Dict[str, Any], top_n: int = 8) -> List[Tuple[str, float]]:
        """
        Get top influencing features for a scenario.
        
        Args:
            scenario: Dictionary with scenario parameters
            top_n: Number of top features to return
        
        Returns:
            List of (feature_name, shap_value) tuples sorted by absolute value
        """
        shap_values = self.get_shap_values(scenario)
        feature_names = self._get_feature_names()
        
        # Create list of (feature, float_value) pairs
        feature_shap_pairs = [(name, float(val)) for name, val in zip(feature_names, shap_values)]
        
        # Sort by absolute SHAP value
        feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top N
        return feature_shap_pairs[:top_n]
    
    def get_shap_summary_data(self, test_data: pd.DataFrame, n_samples: int = 500) -> Dict[str, Any]:
        """
        Get data for SHAP summary (beeswarm) plot.
        
        Args:
            test_data: DataFrame with test samples
            n_samples: Number of samples to use
        
        Returns:
            Dictionary with feature names, SHAP values, and feature values
        """
        if self.explainer is None:
            # Fallback to feature importances
            importances = self.predictor.model.feature_importances_
            feature_names = self._get_feature_names()
            return {
                'feature_names': feature_names,
                'shap_values': np.zeros((n_samples, len(feature_names))),
                'feature_values': np.random.rand(n_samples, len(feature_names)),
                'use_fallback': True
            }
        
        # Sample data
        sample_df = test_data.sample(min(n_samples, len(test_data)), random_state=42)
        
        # Preprocess
        X = self.predictor.preprocessor.transform(sample_df)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For multi-class, use average across classes
        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = np.mean(shap_values, axis=-1)
        
        # Get feature values (normalized)
        feature_values = X.toarray() if hasattr(X, 'toarray') else X
        
        return {
            'feature_names': self._get_feature_names(),
            'shap_values': shap_values,
            'feature_values': feature_values,
            'use_fallback': False
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        numeric_names = self.predictor.preprocessor.numeric_features
        categorical_encoder = self.predictor.preprocessor.preprocessor.named_transformers_['cat']
        categorical_names = categorical_encoder.get_feature_names_out(self.predictor.preprocessor.categorical_features)
        return list(numeric_names) + list(categorical_names)


if __name__ == '__main__':
    # Test explainability
    from src.dataset_generator import generate_dataset
    from src.model import OutcomePredictor
    import joblib
    
    print("Testing SHAP explainability...")
    
    # Load model
    predictor = OutcomePredictor()
    predictor.load('models/random_forest_model.joblib', 'models/preprocessing_pipeline.joblib')
    
    # Generate background data
    df = generate_dataset(1000)
    
    # Initialize explainer
    explainer = ModelExplainer(predictor)
    explainer.initialize_explainer(df, n_background=50)
    
    # Test scenario
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
    
    top_features = explainer.get_top_influencing_features(test_scenario)
    print("\nTop influencing features:")
    for feature, value in top_features:
        print(f"  {feature}: {value:.4f}")
