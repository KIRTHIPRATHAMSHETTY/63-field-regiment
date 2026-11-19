import numpy as np
import pandas as pd
from typing import Optional
import os
from .simulation_engine import simulate_scenario


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: force_ratio, intel_diff, logistics_diff."""
    df = df.copy()
    if 'force_ratio' not in df.columns:
        df['force_ratio'] = df['side_a_force_strength'] / (df['side_b_force_strength'] + 1)
    if 'intel_diff' not in df.columns:
        df['intel_diff'] = df['side_a_intelligence'] - df['side_b_intelligence']
    if 'logistics_diff' not in df.columns:
        df['logistics_diff'] = df['side_a_logistics'] - df['side_b_logistics']
    return df


def generate_dataset(num_samples: int = 20000, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a dataset of simulated military scenarios.
    
    Args:
        num_samples: Number of scenarios to generate
        output_path: Path to save the CSV file (optional)
    
    Returns:
        DataFrame with generated scenarios and outcomes
    """
    np.random.seed(42)
    
    data = []
    terrain_types = ['Open', 'Hilly', 'Forest', 'Desert', 'Urban', 'Mixed']
    weather_types = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Extreme']
    visibility_types = ['High', 'Medium', 'Low']
    time_of_day_types = ['Day', 'Night']
    objective_types = ['Attack', 'Defend', 'Delay', 'Withdraw']
    season_types = ['Spring', 'Summer', 'Autumn', 'Winter']
    roe_types = ['Permissive', 'Restrictive', 'Defensive', 'Offensive']
    
    for i in range(num_samples):
        # Generate random parameters
        config = {
            'side_a_force_strength': np.random.randint(30, 100),
            'side_b_force_strength': np.random.randint(30, 100),
            'side_a_mobility': np.random.uniform(0.3, 1.0),
            'side_b_mobility': np.random.uniform(0.3, 1.0),
            'side_a_logistics': np.random.uniform(0.3, 1.0),
            'side_b_logistics': np.random.uniform(0.3, 1.0),
            'side_a_intelligence': np.random.uniform(0.3, 1.0),
            'side_b_intelligence': np.random.uniform(0.3, 1.0),
            'side_a_communication': np.random.uniform(0.3, 1.0),
            'side_b_communication': np.random.uniform(0.3, 1.0),
            'side_a_readiness': np.random.uniform(0.3, 1.0),
            'side_b_readiness': np.random.uniform(0.3, 1.0),
            'side_a_morale': np.random.uniform(0.3, 1.0),
            'side_b_morale': np.random.uniform(0.3, 1.0),
            'side_a_training': np.random.uniform(0.3, 1.0),
            'side_b_training': np.random.uniform(0.3, 1.0),
            'side_a_supply_distance': np.random.randint(0, 500),
            'side_b_supply_distance': np.random.randint(0, 500),
            'side_a_surprise': np.random.uniform(0.0, 1.0),
            'side_b_surprise': np.random.uniform(0.0, 1.0),
            'side_a_fatigue': np.random.uniform(0.0, 1.0),
            'side_b_fatigue': np.random.uniform(0.0, 1.0),
            'side_a_equipment_quality': np.random.uniform(0.3, 1.0),
            'side_b_equipment_quality': np.random.uniform(0.3, 1.0),
            'electronic_warfare_level': np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0]),
            'objective_type': np.random.choice(objective_types),
            'season': np.random.choice(season_types),
            'rules_of_engagement': np.random.choice(roe_types),
            'terrain_type': np.random.choice(terrain_types),
            'weather': np.random.choice(weather_types),
            'visibility': np.random.choice(visibility_types),
            'time_of_day': np.random.choice(time_of_day_types),
            'environmental_difficulty': np.random.uniform(0.0, 1.0)
        }
        
        # Simulate the scenario
        result = simulate_scenario(config)
        
        # Create row
        row = {
            'scenario_id': f'SCN-{i:05d}',
            'side_a_force_strength': config['side_a_force_strength'],
            'side_b_force_strength': config['side_b_force_strength'],
            'side_a_mobility': config['side_a_mobility'],
            'side_b_mobility': config['side_b_mobility'],
            'side_a_logistics': config['side_a_logistics'],
            'side_b_logistics': config['side_b_logistics'],
            'side_a_intelligence': config['side_a_intelligence'],
            'side_b_intelligence': config['side_b_intelligence'],
            'side_a_communication': config['side_a_communication'],
            'side_b_communication': config['side_b_communication'],
            'side_a_readiness': config['side_a_readiness'],
            'side_b_readiness': config['side_b_readiness'],
            'side_a_morale': config['side_a_morale'],
            'side_b_morale': config['side_b_morale'],
            'side_a_training': config['side_a_training'],
            'side_b_training': config['side_b_training'],
            'side_a_supply_distance': config['side_a_supply_distance'],
            'side_b_supply_distance': config['side_b_supply_distance'],
            'side_a_surprise': config['side_a_surprise'],
            'side_b_surprise': config['side_b_surprise'],
            'side_a_fatigue': config['side_a_fatigue'],
            'side_b_fatigue': config['side_b_fatigue'],
            'side_a_equipment_quality': config['side_a_equipment_quality'],
            'side_b_equipment_quality': config['side_b_equipment_quality'],
            'electronic_warfare_level': config['electronic_warfare_level'],
            'objective_type': config['objective_type'],
            'season': config['season'],
            'rules_of_engagement': config['rules_of_engagement'],
            'terrain_type': config['terrain_type'],
            'weather': config['weather'],
            'visibility': config['visibility'],
            'time_of_day': config['time_of_day'],
            'environmental_difficulty': config['environmental_difficulty'],
            'outcome': result['outcome']
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df['force_ratio'] = df['side_a_force_strength'] / (df['side_b_force_strength'] + 1)
    df['intel_diff'] = df['side_a_intelligence'] - df['side_b_intelligence']
    df['logistics_diff'] = df['side_a_logistics'] - df['side_b_logistics']
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return df


if __name__ == '__main__':
    # Generate dataset
    print("Generating dataset with 20,000 scenarios...")
    df = generate_dataset(20000, 'data/barbarik_dataset_new.csv')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['outcome'].value_counts(normalize=True))
    print(f"\nFirst few rows:")
    print(df.head())
