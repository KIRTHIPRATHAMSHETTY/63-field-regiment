import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os


class DataPreprocessor:
    """Handle data preprocessing for the ML model."""
    
    def __init__(self):
        self.numeric_features = [
            'side_a_force_strength', 'side_b_force_strength',
            'side_a_mobility', 'side_b_mobility',
            'side_a_logistics', 'side_b_logistics',
            'side_a_intelligence', 'side_b_intelligence',
            'side_a_communication', 'side_b_communication',
            'side_a_readiness', 'side_b_readiness',
            'side_a_morale', 'side_b_morale',
            'side_a_training', 'side_b_training',
            'side_a_supply_distance', 'side_b_supply_distance',
            'side_a_surprise', 'side_b_surprise',
            'side_a_fatigue', 'side_b_fatigue',
            'side_a_equipment_quality', 'side_b_equipment_quality',
            'electronic_warfare_level', 'environmental_difficulty',
            'force_ratio', 'intel_diff', 'logistics_diff'
        ]
        
        self.categorical_features = [
            'terrain_type', 'weather', 'visibility', 'time_of_day',
            'objective_type', 'season', 'rules_of_engagement'
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
    
    def _ensure_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure derived features exist in DataFrame."""
        df = df.copy()
        if 'force_ratio' not in df.columns and 'side_a_force_strength' in df.columns:
            df['force_ratio'] = df['side_a_force_strength'] / (df['side_b_force_strength'] + 1)
        if 'intel_diff' not in df.columns and 'side_a_intelligence' in df.columns:
            df['intel_diff'] = df['side_a_intelligence'] - df['side_b_intelligence']
        if 'logistics_diff' not in df.columns and 'side_a_logistics' in df.columns:
            df['logistics_diff'] = df['side_a_logistics'] - df['side_b_logistics']
        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit the preprocessor on training data only and transform both train and test.
        
        Args:
            df: Input DataFrame with features and outcome column
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        df = self._ensure_derived_features(df)
        X = df.drop(columns=['scenario_id', 'outcome'], errors='ignore')
        y = df['outcome']
        
        # Stratified 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit on training data ONLY and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame with features
        
        Returns:
            Processed features
        """
        df = self._ensure_derived_features(df)
        X = df.drop(columns=['scenario_id', 'outcome'], errors='ignore')
        return self.preprocessor.transform(X)
    
    def save(self, path: str):
        """Save the preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
    
    def load(self, path: str):
        """Load the preprocessor from disk."""
        self.preprocessor = joblib.load(path)


if __name__ == '__main__':
    # Test preprocessing
    from .dataset_generator import generate_dataset
    
    print("Testing preprocessing pipeline...")
    df = generate_dataset(1000)  # Small test
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))
    print(f"\nTest class distribution:")
    print(y_test.value_counts(normalize=True))
