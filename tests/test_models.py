"""Model tests."""

import numpy as np
import pytest

from app.ml_model import MLModel


def test_ml_model_initialization():
    """Test ML model initialization."""
    model = MLModel()
    assert model.model is None
    assert model.model_score == 0.0
    assert model.is_loaded is False


def test_ml_model_predict_without_loading():
    """Test prediction fails without loading model."""
    model = MLModel()
    features = np.array([900.0, 15.0, 120.0, 1020.0])
    
    with pytest.raises(ValueError, match="Model not loaded"):
        model.predict(features)


def test_ml_model_get_info_not_loaded():
    """Test getting model info when not loaded."""
    model = MLModel()
    info = model.get_model_info()
    
    assert info["loaded"] is False
    assert info["coefficients"] is None
    assert info["intercept"] is None

# Made with Bob
