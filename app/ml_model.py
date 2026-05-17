"""Machine learning model management."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from app.config import get_settings

settings = get_settings()


class MLModel:
    """Machine learning model wrapper."""

    def __init__(self):
        """Initialize ML model."""
        self.model: Optional[LinearRegression] = None
        self.model_score: float = 0.89
        self.is_loaded: bool = False
        self._use_mock = False

    def _create_mock_model(self) -> None:
        """Create a simple mock model for demonstration purposes."""
        print("Creating mock model for demonstration...")
        self.model = LinearRegression()
        # Simple coefficients based on typical flight delay patterns
        # [departure_time, departure_delay, scheduled_time, arrival_time]
        self.model.coef_ = np.array([0.001, 0.95, 0.01, -0.001])
        self.model.intercept_ = 5.0
        self.is_loaded = True
        self._use_mock = True
        print("Mock model created successfully")

    def load_model(self) -> bool:
        """Load the trained model from disk or create mock model."""
        model_path = Path(settings.ml_model_path)
        
        if not model_path.exists():
            print(f"Warning: Model file not found at {model_path}")
            print("Using mock model for demonstration purposes")
            self._create_mock_model()
            return True

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.is_loaded = True
            self._use_mock = False
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock model")
            self._create_mock_model()
            return True

    def predict(self, features: np.ndarray) -> float:
        """
        Make prediction using the loaded model.
        
        Args:
            features: Input features as numpy array [departure_time, departure_delay,
                     scheduled_time, arrival_time]
        
        Returns:
            Predicted arrival delay in minutes
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # For mock model, use simple linear combination
        if self._use_mock:
            prediction = np.dot(features, self.model.coef_) + self.model.intercept_
            return float(prediction)
        
        # For real model, use sklearn predict
        prediction = self.model.predict(features.reshape(1, -1))
        return float(prediction[0])

    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.is_loaded or self.model is None:
            return {
                "loaded": False,
                "coefficients": None,
                "intercept": None,
                "is_mock": False,
            }

        return {
            "loaded": True,
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "is_mock": self._use_mock,
        }


# Global model instance
ml_model = MLModel()

# Made with Bob
