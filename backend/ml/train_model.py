"""
Train bearing health classifier on NASA IMS dataset.

Uses XGBoost to classify bearing health into three states:
- HEALTHY: Normal operation
- DEGRADING: Early-stage fault development
- CRITICAL: Imminent failure

Labeling strategy:
- Last 10% of run-to-failure: CRITICAL
- 10-30% from end: DEGRADING
- First 70%: HEALTHY
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from .feature_extraction import extract_features_from_file, get_feature_names, load_nasa_file, extract_features

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = Path(__file__).parent / "models"


def get_test_directories() -> List[Path]:
    """Find all test directories in the raw data."""
    tests = []
    for name in ['1st_test', '2nd_test', '3rd_test']:
        test_dir = RAW_DIR / name
        if test_dir.exists():
            tests.append(test_dir)
    return tests


def get_data_files(test_dir: Path) -> List[Path]:
    """Get all data files from a test directory, sorted chronologically."""
    files = []
    for f in test_dir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            files.append(f)

    # Sort by filename (which contains timestamp)
    files.sort(key=lambda x: x.name)
    return files


def label_files(files: List[Path]) -> List[str]:
    """
    Assign health labels based on position in run-to-failure sequence.

    Labeling strategy:
    - First 70%: HEALTHY
    - 70-90%: DEGRADING
    - Last 10%: CRITICAL
    """
    n = len(files)
    labels = []

    for i in range(n):
        progress = i / n
        if progress < 0.70:
            labels.append('HEALTHY')
        elif progress < 0.90:
            labels.append('DEGRADING')
        else:
            labels.append('CRITICAL')

    return labels


def process_test_data(test_dir: Path, channel: int = 0, sample_every: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single test directory into features and labels.

    Args:
        test_dir: Path to test directory
        channel: Which bearing channel to extract (failed bearing)
        sample_every: Process every Nth file to reduce computation

    Returns:
        features: (n_samples, n_features) array
        labels: (n_samples,) array of string labels
    """
    print(f"\nProcessing {test_dir.name}...")
    files = get_data_files(test_dir)

    if not files:
        print(f"  No files found in {test_dir}")
        return np.array([]), np.array([])

    print(f"  Found {len(files)} files")

    # Sample files to speed up training
    sampled_files = files[::sample_every]
    labels = label_files(files)[::sample_every]

    print(f"  Processing {len(sampled_files)} files (every {sample_every}th)")

    features_list = []
    valid_labels = []

    for i, (filepath, label) in enumerate(zip(sampled_files, labels)):
        try:
            feats = extract_features_from_file(str(filepath), channel)
            features_list.append(list(feats.values()))
            valid_labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(sampled_files)} files")

        except Exception as e:
            print(f"    Error processing {filepath.name}: {e}")
            continue

    if not features_list:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(valid_labels)


def build_dataset(sample_every: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build complete dataset from all available tests.

    The NASA IMS dataset has 3 tests with different failing bearings:
    - Test 1: Bearing 3 (inner race failure) - channel 2
    - Test 2: Bearing 1 (outer race failure) - channel 0
    - Test 3: Bearing 3 (outer race failure) - channel 2
    """
    all_features = []
    all_labels = []

    test_configs = [
        ('1st_test', 2),  # Bearing 3 failed
        ('2nd_test', 0),  # Bearing 1 failed
        ('3rd_test', 2),  # Bearing 3 failed
    ]

    for test_name, channel in test_configs:
        test_dir = RAW_DIR / test_name
        if test_dir.exists():
            features, labels = process_test_data(test_dir, channel, sample_every)
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)

    if not all_features:
        raise ValueError("No data found! Run download_nasa_data.py first.")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    print(f"\nTotal dataset: {len(X)} samples")
    print(f"Label distribution:")
    for label in ['HEALTHY', 'DEGRADING', 'CRITICAL']:
        count = np.sum(y == label)
        print(f"  {label}: {count} ({100 * count / len(y):.1f}%)")

    return X, y


def train_classifier(X: np.ndarray, y: np.ndarray) -> Tuple[XGBClassifier, StandardScaler, Dict]:
    """
    Train XGBoost classifier with proper train/test split.

    Returns:
        model: Trained XGBoost classifier
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
    """
    # Encode labels
    label_map = {'HEALTHY': 0, 'DEGRADING': 1, 'CRITICAL': 2}
    y_encoded = np.array([label_map[label] for label in y])

    # Train/test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_labels = [['HEALTHY', 'DEGRADING', 'CRITICAL'][i] for i in y_pred]
    y_test_labels = [['HEALTHY', 'DEGRADING', 'CRITICAL'][i] for i in y_test]

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=['HEALTHY', 'DEGRADING', 'CRITICAL'])
    print(cm)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': cm.tolist(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_names': get_feature_names(),
    }

    return model, scaler, metrics


def save_model(model: XGBClassifier, scaler: StandardScaler, metrics: Dict):
    """Save trained model, scaler, and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "bearing_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    # Save scaler
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
        'model_type': 'XGBClassifier',
        'labels': ['HEALTHY', 'DEGRADING', 'CRITICAL'],
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Bearing Health Classifier Training")
    print("=" * 60)

    # Try to use NASA data first, fall back to synthetic
    use_synthetic = False

    if RAW_DIR.exists():
        # Check for actual test directories
        test_dirs = [RAW_DIR / name for name in ['1st_test', '2nd_test', '3rd_test']]
        if any(d.exists() for d in test_dirs):
            print("\nUsing NASA IMS Bearing Dataset")
            try:
                X, y = build_dataset(sample_every=5)
            except Exception as e:
                print(f"Error loading NASA data: {e}")
                use_synthetic = True
        else:
            use_synthetic = True
    else:
        use_synthetic = True

    if use_synthetic:
        print("\nNASA data not available - generating synthetic training data")
        print("(Based on bearing failure physics from NASA IMS research)")
        from .generate_synthetic_data import generate_training_dataset
        X, y = generate_training_dataset(n_samples_per_class=500)

    # Train model
    model, scaler, metrics = train_classifier(X, y)

    # Save everything
    save_model(model, scaler, metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 (macro): {metrics['f1_macro']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
