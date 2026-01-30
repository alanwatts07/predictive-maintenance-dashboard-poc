"""
Bearing health predictor using trained ML model.

Loads the trained XGBoost model and provides real-time inference
for vibration data received from sensors.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Add ml directory to path for imports
ML_DIR = Path(__file__).parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from feature_extraction import extract_features, get_feature_names

MODELS_DIR = ML_DIR / "models"


class BearingPredictor:
    """
    Real-time bearing health predictor.

    Uses trained XGBoost model to classify bearing health state
    from vibration signal features.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.labels = ['HEALTHY', 'DEGRADING', 'CRITICAL']
        self.feature_names = get_feature_names()
        self._load_model()

    def _load_model(self):
        """Load trained model and scaler."""
        model_path = MODELS_DIR / "bearing_classifier.joblib"
        scaler_path = MODELS_DIR / "feature_scaler.joblib"

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            print("Run 'python ml/train_model.py' to train the model.")
            return

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Loaded bearing health model from {model_path}")

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None and self.scaler is not None

    def predict_from_signal(self, signal: np.ndarray, sample_rate: float = 20000) -> Dict:
        """
        Predict bearing health from raw vibration signal.

        Args:
            signal: 1D numpy array of vibration data
            sample_rate: Sampling rate in Hz

        Returns:
            Dictionary with:
                - state: 'HEALTHY', 'DEGRADING', or 'CRITICAL'
                - confidence: Probability of predicted class
                - probabilities: Dict of all class probabilities
                - degradation: 0.0-1.0 degradation score
                - features: Extracted feature values
        """
        if not self.is_ready():
            return {
                'state': 'UNKNOWN',
                'confidence': 0.0,
                'probabilities': {},
                'degradation': 0.0,
                'features': {},
                'error': 'Model not loaded'
            }

        # Extract features
        features = extract_features(signal, sample_rate)
        feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)

        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)

        # Predict
        probabilities = self.model.predict_proba(feature_scaled)[0]
        predicted_class = np.argmax(probabilities)
        predicted_label = self.labels[predicted_class]
        confidence = probabilities[predicted_class]

        # Compute degradation score (0 = healthy, 1 = critical)
        # Weighted sum: HEALTHY=0, DEGRADING=0.5, CRITICAL=1.0
        degradation = probabilities[1] * 0.5 + probabilities[2] * 1.0

        return {
            'state': predicted_label,
            'confidence': float(confidence),
            'probabilities': {
                'HEALTHY': float(probabilities[0]),
                'DEGRADING': float(probabilities[1]),
                'CRITICAL': float(probabilities[2]),
            },
            'degradation': float(degradation),
            'features': features,
        }

    def predict_from_features(self, features: Dict[str, float]) -> Dict:
        """
        Predict bearing health from pre-extracted features.

        Useful when features are computed externally or for testing.
        """
        if not self.is_ready():
            return {
                'state': 'UNKNOWN',
                'confidence': 0.0,
                'probabilities': {},
                'degradation': 0.0,
                'error': 'Model not loaded'
            }

        feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vector)

        probabilities = self.model.predict_proba(feature_scaled)[0]
        predicted_class = np.argmax(probabilities)
        predicted_label = self.labels[predicted_class]
        confidence = probabilities[predicted_class]

        degradation = probabilities[1] * 0.5 + probabilities[2] * 1.0

        return {
            'state': predicted_label,
            'confidence': float(confidence),
            'probabilities': {
                'HEALTHY': float(probabilities[0]),
                'DEGRADING': float(probabilities[1]),
                'CRITICAL': float(probabilities[2]),
            },
            'degradation': float(degradation),
        }


# Singleton instance
_predictor: Optional[BearingPredictor] = None


def get_predictor() -> BearingPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = BearingPredictor()
    return _predictor


# Convenience functions
def predict(signal: np.ndarray, sample_rate: float = 20000) -> Dict:
    """Predict bearing health from raw vibration signal."""
    return get_predictor().predict_from_signal(signal, sample_rate)


def is_model_ready() -> bool:
    """Check if the ML model is loaded and ready."""
    return get_predictor().is_ready()


if __name__ == "__main__":
    # Test with synthetic data
    predictor = BearingPredictor()

    if predictor.is_ready():
        # Generate test signals
        t = np.linspace(0, 1, 20000)

        # Healthy signal
        healthy_signal = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
        result = predictor.predict_from_signal(healthy_signal)
        print(f"\nHealthy signal prediction:")
        print(f"  State: {result['state']} (confidence: {result['confidence']:.2%})")
        print(f"  Degradation: {result['degradation']:.2%}")

        # Faulty signal
        faulty_signal = 2.0 * np.sin(2 * np.pi * 100 * t) + 0.8 * np.random.randn(len(t))
        impulses = np.random.choice(len(t), 100, replace=False)
        faulty_signal[impulses] += 8 * np.random.randn(100)
        result = predictor.predict_from_signal(faulty_signal)
        print(f"\nFaulty signal prediction:")
        print(f"  State: {result['state']} (confidence: {result['confidence']:.2%})")
        print(f"  Degradation: {result['degradation']:.2%}")
    else:
        print("Model not loaded. Run train_model.py first.")
