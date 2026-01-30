"""
Feature extraction for vibration signals.

Extracts statistical and spectral features commonly used in bearing fault detection:
- Time domain: RMS, peak, crest factor, kurtosis, skewness
- Frequency domain: spectral centroid, spectral spread, band energies
"""

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Dict, List, Tuple
import os
from pathlib import Path


def compute_rms(signal: np.ndarray) -> float:
    """Root Mean Square - overall vibration energy."""
    return np.sqrt(np.mean(signal ** 2))


def compute_peak(signal: np.ndarray) -> float:
    """Peak amplitude."""
    return np.max(np.abs(signal))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Crest factor = Peak / RMS. High values indicate impulsive events."""
    rms = compute_rms(signal)
    if rms == 0:
        return 0
    return compute_peak(signal) / rms


def compute_kurtosis(signal: np.ndarray) -> float:
    """Kurtosis - measure of tailedness/impulsiveness. High = shock events."""
    return stats.kurtosis(signal, fisher=True)


def compute_skewness(signal: np.ndarray) -> float:
    """Skewness - asymmetry of distribution."""
    return stats.skew(signal)


def compute_std(signal: np.ndarray) -> float:
    """Standard deviation."""
    return np.std(signal)


def compute_clearance_factor(signal: np.ndarray) -> float:
    """Clearance factor - sensitive to early bearing faults."""
    peak = compute_peak(signal)
    mean_sqrt = np.mean(np.sqrt(np.abs(signal))) ** 2
    if mean_sqrt == 0:
        return 0
    return peak / mean_sqrt


def compute_shape_factor(signal: np.ndarray) -> float:
    """Shape factor = RMS / mean(|signal|)."""
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0:
        return 0
    return compute_rms(signal) / mean_abs


def compute_impulse_factor(signal: np.ndarray) -> float:
    """Impulse factor = Peak / mean(|signal|)."""
    mean_abs = np.mean(np.abs(signal))
    if mean_abs == 0:
        return 0
    return compute_peak(signal) / mean_abs


def compute_spectral_features(signal: np.ndarray, sample_rate: float = 20000) -> Dict[str, float]:
    """
    Compute frequency domain features.

    Args:
        signal: Time-domain vibration signal
        sample_rate: Sampling rate in Hz (NASA data is 20kHz)

    Returns:
        Dictionary of spectral features
    """
    n = len(signal)

    # FFT
    fft_vals = fft(signal)
    fft_mag = np.abs(fft_vals[:n // 2])
    freqs = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    # Normalize magnitude spectrum
    fft_mag_norm = fft_mag / (np.sum(fft_mag) + 1e-10)

    # Spectral centroid (center of mass of spectrum)
    spectral_centroid = np.sum(freqs * fft_mag_norm)

    # Spectral spread (variance around centroid)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag_norm))

    # Band energies (useful for bearing fault frequencies)
    # Low: 0-2kHz, Mid: 2-5kHz, High: 5-10kHz
    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(fft_mag[mask] ** 2)

    total_energy = np.sum(fft_mag ** 2) + 1e-10

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'band_energy_low': band_energy(0, 2000) / total_energy,
        'band_energy_mid': band_energy(2000, 5000) / total_energy,
        'band_energy_high': band_energy(5000, 10000) / total_energy,
    }


def extract_features(signal: np.ndarray, sample_rate: float = 20000) -> Dict[str, float]:
    """
    Extract all features from a vibration signal.

    Args:
        signal: 1D numpy array of vibration data
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary of feature names to values
    """
    features = {
        'rms': compute_rms(signal),
        'peak': compute_peak(signal),
        'crest_factor': compute_crest_factor(signal),
        'kurtosis': compute_kurtosis(signal),
        'skewness': compute_skewness(signal),
        'std': compute_std(signal),
        'clearance_factor': compute_clearance_factor(signal),
        'shape_factor': compute_shape_factor(signal),
        'impulse_factor': compute_impulse_factor(signal),
    }

    # Add spectral features
    spectral = compute_spectral_features(signal, sample_rate)
    features.update(spectral)

    return features


def load_nasa_file(filepath: str) -> np.ndarray:
    """
    Load a NASA bearing data file.

    NASA IMS format: ASCII file with 4 or 8 columns (one per bearing channel)
    Each row is a sample, ~20480 samples per file (1 second at 20kHz)
    """
    data = np.loadtxt(filepath, delimiter='\t' if '\t' in open(filepath).read(1000) else None)
    return data


def extract_features_from_file(filepath: str, channel: int = 0) -> Dict[str, float]:
    """
    Extract features from a NASA bearing data file.

    Args:
        filepath: Path to the data file
        channel: Which bearing channel to use (0-3 or 0-7)

    Returns:
        Feature dictionary
    """
    data = load_nasa_file(filepath)

    # Handle both 1D and 2D data
    if data.ndim == 1:
        signal = data
    else:
        signal = data[:, channel]

    return extract_features(signal)


def get_feature_names() -> List[str]:
    """Return ordered list of feature names."""
    return [
        'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness',
        'std', 'clearance_factor', 'shape_factor', 'impulse_factor',
        'spectral_centroid', 'spectral_spread',
        'band_energy_low', 'band_energy_mid', 'band_energy_high'
    ]


if __name__ == "__main__":
    # Test with synthetic data
    t = np.linspace(0, 1, 20000)

    # Healthy signal: low amplitude, smooth
    healthy = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))

    # Faulty signal: higher amplitude, impulsive
    faulty = 2.0 * np.sin(2 * np.pi * 100 * t) + 0.5 * np.random.randn(len(t))
    # Add impulses
    impulse_indices = np.random.choice(len(t), 50, replace=False)
    faulty[impulse_indices] += 5 * np.random.randn(50)

    print("Healthy bearing features:")
    for k, v in extract_features(healthy).items():
        print(f"  {k}: {v:.4f}")

    print("\nFaulty bearing features:")
    for k, v in extract_features(faulty).items():
        print(f"  {k}: {v:.4f}")
