"""
Generate synthetic bearing vibration data for training.

Creates data that mimics the statistical properties of the NASA IMS bearing dataset
when the actual dataset is unavailable.

The synthetic data models:
- Healthy bearings: Low RMS, low kurtosis, stable spectrum
- Degrading bearings: Increasing RMS, periodic defect impacts
- Critical bearings: High RMS, shock events, broadband noise
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import json

from .feature_extraction import extract_features, get_feature_names

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def generate_healthy_signal(n_samples: int = 20000, sample_rate: float = 20000) -> np.ndarray:
    """
    Generate synthetic healthy bearing vibration signal.

    Characteristics:
    - Low baseline RMS (~0.3-0.6g)
    - Low kurtosis (near Gaussian)
    - Clean spectral content (shaft frequency + harmonics)
    """
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    # Shaft frequency (typical: 30-50 Hz for 1800-3000 RPM)
    shaft_freq = 33.3  # ~2000 RPM

    # Base signal: shaft rotation + minor harmonics
    signal = (
        0.3 * np.sin(2 * np.pi * shaft_freq * t) +
        0.1 * np.sin(2 * np.pi * 2 * shaft_freq * t) +  # 2x harmonic
        0.05 * np.sin(2 * np.pi * 3 * shaft_freq * t)   # 3x harmonic
    )

    # Add Gaussian noise (low level)
    noise = np.random.normal(0, 0.08, n_samples)

    return signal + noise


def generate_degrading_signal(n_samples: int = 20000, sample_rate: float = 20000,
                              severity: float = 0.5) -> np.ndarray:
    """
    Generate synthetic degrading bearing vibration signal.

    Characteristics:
    - Elevated RMS (0.8-1.5g)
    - Periodic impulses at defect frequency
    - Increased kurtosis
    """
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    shaft_freq = 33.3
    # Ball Pass Frequency Outer race (BPFO) - typical defect frequency
    bpfo = shaft_freq * 3.5  # ~116 Hz for typical bearing geometry

    # Base signal with elevated amplitude
    signal = (
        (0.4 + 0.3 * severity) * np.sin(2 * np.pi * shaft_freq * t) +
        (0.15 + 0.1 * severity) * np.sin(2 * np.pi * 2 * shaft_freq * t) +
        0.08 * np.sin(2 * np.pi * 3 * shaft_freq * t)
    )

    # Add defect frequency modulation
    defect_mod = (0.2 + 0.3 * severity) * np.sin(2 * np.pi * bpfo * t)

    # Add periodic impulses (bearing impacts)
    impulse_period = int(sample_rate / bpfo)
    impulses = np.zeros(n_samples)
    for i in range(0, n_samples, impulse_period):
        if i < n_samples:
            # Exponentially decaying impulse
            decay_len = min(100, n_samples - i)
            impulse = (0.3 + 0.5 * severity) * np.exp(-np.arange(decay_len) / 20)
            impulses[i:i+decay_len] += impulse * np.random.choice([-1, 1])

    # Add noise (increased level)
    noise = np.random.normal(0, 0.12 + 0.08 * severity, n_samples)

    return signal + defect_mod + impulses + noise


def generate_critical_signal(n_samples: int = 20000, sample_rate: float = 20000) -> np.ndarray:
    """
    Generate synthetic critical/failing bearing vibration signal.

    Characteristics:
    - High RMS (2-5g)
    - High kurtosis (shock events)
    - Broadband noise increase
    - Random high-amplitude impacts
    """
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    shaft_freq = 33.3
    bpfo = shaft_freq * 3.5
    bpfi = shaft_freq * 5.2  # Ball Pass Frequency Inner race

    # Elevated base signal
    signal = (
        0.8 * np.sin(2 * np.pi * shaft_freq * t) +
        0.4 * np.sin(2 * np.pi * 2 * shaft_freq * t) +
        0.2 * np.sin(2 * np.pi * 3 * shaft_freq * t)
    )

    # Multiple defect frequencies (outer and inner race)
    defect = (
        0.6 * np.sin(2 * np.pi * bpfo * t) +
        0.4 * np.sin(2 * np.pi * bpfi * t)
    )

    # Heavy periodic impulses
    impulse_period = int(sample_rate / bpfo)
    impulses = np.zeros(n_samples)
    for i in range(0, n_samples, impulse_period):
        if i < n_samples:
            decay_len = min(150, n_samples - i)
            impulse = 1.2 * np.exp(-np.arange(decay_len) / 30)
            impulses[i:i+decay_len] += impulse * np.random.choice([-1, 1])

    # Random shock events (metal-to-metal contact)
    n_shocks = np.random.randint(20, 50)
    shock_indices = np.random.choice(n_samples, n_shocks, replace=False)
    for idx in shock_indices:
        shock_len = min(50, n_samples - idx)
        shock = np.random.uniform(2, 4) * np.exp(-np.arange(shock_len) / 10)
        impulses[idx:idx+shock_len] += shock * np.random.choice([-1, 1])

    # High noise level (broadband)
    noise = np.random.normal(0, 0.3, n_samples)

    return signal + defect + impulses + noise


def generate_training_dataset(n_samples_per_class: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete training dataset with balanced classes.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels array
    """
    print(f"Generating {n_samples_per_class} samples per class...")

    features_list = []
    labels = []

    # Generate HEALTHY samples
    print("  Generating HEALTHY samples...")
    for i in range(n_samples_per_class):
        # Add some variation
        signal = generate_healthy_signal()
        # Random amplitude scaling (different bearing sizes/conditions)
        scale = np.random.uniform(0.8, 1.2)
        features = extract_features(signal * scale)
        features_list.append(list(features.values()))
        labels.append('HEALTHY')

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_samples_per_class}")

    # Generate DEGRADING samples (varying severity)
    print("  Generating DEGRADING samples...")
    for i in range(n_samples_per_class):
        severity = np.random.uniform(0.2, 0.8)  # Varying degradation levels
        signal = generate_degrading_signal(severity=severity)
        scale = np.random.uniform(0.8, 1.2)
        features = extract_features(signal * scale)
        features_list.append(list(features.values()))
        labels.append('DEGRADING')

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_samples_per_class}")

    # Generate CRITICAL samples
    print("  Generating CRITICAL samples...")
    for i in range(n_samples_per_class):
        signal = generate_critical_signal()
        scale = np.random.uniform(0.9, 1.1)
        features = extract_features(signal * scale)
        features_list.append(list(features.values()))
        labels.append('CRITICAL')

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_samples_per_class}")

    X = np.array(features_list)
    y = np.array(labels)

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"\nGenerated {len(X)} total samples")
    return X, y


def save_synthetic_dataset(X: np.ndarray, y: np.ndarray):
    """Save the synthetic dataset for future use."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.save(PROCESSED_DIR / "X_synthetic.npy", X)
    np.save(PROCESSED_DIR / "y_synthetic.npy", y)

    # Also save as CSV for inspection
    df = pd.DataFrame(X, columns=get_feature_names())
    df['label'] = y
    df.to_csv(PROCESSED_DIR / "synthetic_dataset.csv", index=False)

    print(f"Saved dataset to {PROCESSED_DIR}")


def load_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load previously generated synthetic dataset."""
    X = np.load(PROCESSED_DIR / "X_synthetic.npy")
    y = np.load(PROCESSED_DIR / "y_synthetic.npy")
    return X, y


if __name__ == "__main__":
    X, y = generate_training_dataset(n_samples_per_class=500)
    save_synthetic_dataset(X, y)

    print("\nDataset statistics:")
    for label in ['HEALTHY', 'DEGRADING', 'CRITICAL']:
        mask = y == label
        print(f"\n{label}:")
        print(f"  RMS mean: {X[mask, 0].mean():.3f} ± {X[mask, 0].std():.3f}")
        print(f"  Kurtosis mean: {X[mask, 3].mean():.3f} ± {X[mask, 3].std():.3f}")
