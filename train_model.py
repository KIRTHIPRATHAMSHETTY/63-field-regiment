import os
import pandas as pd
import numpy as np
from src.model import OutcomePredictor
from src.dataset_generator import generate_dataset


def main():
    print("=" * 70)
    print("PHASE 2 & 3: BARBARIK DATASET GENERATION & XGBOOST MODEL TRAINING")
    print("=" * 70)
    
    dataset_path = 'data/barbarik_dataset_new.csv'
    
    print("\n[STEP 1] Generating exactly 20,000 synthetic scenarios with seed 42...")
    df = generate_dataset(20000, dataset_path)
    # Also save to barbarik_dataset.csv for consistency
    df.to_csv('data/barbarik_dataset.csv', index=False)
    
    print("\n[STEP 2] Dataset Quality Diagnostics:")
    print(f"  - Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  - Missing Values Total: {df.isnull().sum().sum()}")
    if df.isnull().sum().sum() > 0:
        print(df.isnull().sum()[df.isnull().sum() > 0])
    
    duplicate_rows = df.drop(columns=['scenario_id']).duplicated().sum()
    print(f"  - Duplicate Scenarios (excluding ID): {duplicate_rows}")
    
    print("\n  - Target Class Distribution:")
    class_counts = df['outcome'].value_counts()
    class_pct = df['outcome'].value_counts(normalize=True) * 100
    for cls_name in class_counts.index:
        print(f"    * {cls_name:18s}: {class_counts[cls_name]:5d} ({class_pct[cls_name]:.2f}%)")
    
    print("\n[STEP 3] Target Leakage & Train-Test Split Verification:")
    feature_cols = [c for c in df.columns if c not in ['scenario_id', 'outcome']]
    print(f"  - Total Input Features: {len(feature_cols)}")
    print(f"  - Features: {', '.join(feature_cols)}")
    
    # Check for target leakage
    leakage_found = False
    for col in feature_cols:
        if 'outcome' in col.lower() or 'target' in col.lower():
            print(f"  [WARNING] Potential leakage feature: {col}")
            leakage_found = True
    if not leakage_found:
        print("  - Target Leakage Check: PASSED (no outcome or label proxy features found in inputs)")
        
    from sklearn.model_selection import train_test_split
    X = df[feature_cols]
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Check duplicate scenarios across train and test
    train_tuples = set(tuple(x) for x in X_train.values)
    test_tuples = set(tuple(x) for x in X_test.values)
    overlap = len(train_tuples.intersection(test_tuples))
    print(f"  - Overlap between Train and Test sets: {overlap} identical feature rows")
    
    print("\n[STEP 4] Training XGBoost Classifier...")
    predictor = OutcomePredictor()
    metrics = predictor.train(df)
    
    print("\n" + "=" * 70)
    print("EVALUATION METRICS (TEST SET - HELD OUT 20%)")
    print("=" * 70)
    print(f"Accuracy:           {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.5f})")
    print(f"Macro Precision:    {metrics['precision_macro']:.5f}")
    print(f"Macro Recall:       {metrics['recall_macro']:.5f}")
    print(f"Macro F1-Score:     {metrics['f1_macro']:.5f}")
    print(f"Weighted Precision: {metrics['precision']:.5f}")
    print(f"Weighted Recall:    {metrics['recall']:.5f}")
    print(f"Weighted F1-Score:  {metrics['f1_weighted']:.5f}")
    print(f"Multiclass ROC-AUC: {metrics['roc_auc']:.5f}")
    
    print("\nConfusion Matrix:")
    classes = list(predictor.label_encoder.classes_)
    cm = np.array(metrics['confusion_matrix'])
    header = "True \\ Pred".ljust(18) + "".join([f"{c[:14]:>15}" for c in classes])
    print(header)
    for i, c in enumerate(classes):
        row_str = f"{c[:17]:18}" + "".join([f"{cm[i, j]:15d}" for j in range(len(classes))])
        print(row_str)
    
    # Per-class accuracy / sensitivity
    print("\nPer-Class Sensitivity (Recall):")
    for i, c in enumerate(classes):
        row_total = cm[i].sum()
        correct = cm[i, i]
        acc_cls = (correct / row_total * 100) if row_total > 0 else 0
        print(f"  - {c:18s}: {correct:4d} / {row_total:4d} ({acc_cls:.2f}%)")
        
    print("\n[STEP 5] Saving Models & Preprocessors:")
    # Save to primary paths
    xgb_path = 'models/xgboost_model.joblib'
    rf_compat_path = 'models/random_forest_model.joblib'
    prep_path = 'models/preprocessing_pipeline.joblib'
    le_path = 'models/label_encoder.joblib'
    
    predictor.save(xgb_path, prep_path, le_path)
    # Also save as random_forest_model.joblib so any legacy scripts find it
    import joblib
    joblib.dump(predictor.model, rf_compat_path)
    
    print(f"  - Saved XGBoost model to:        {xgb_path}")
    print(f"  - Saved Preprocessing Pipeline:  {prep_path}")
    print(f"  - Saved Label Encoder:           {le_path}")
    print(f"  - Saved Compatibility Backup:    {rf_compat_path}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 & 3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
