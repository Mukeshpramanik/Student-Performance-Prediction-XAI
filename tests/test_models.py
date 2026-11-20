"""
Tests for regression and classification model training, metrics, and persistence.
"""

import pytest
import joblib
from pathlib import Path
from config.settings import MODELS_DIR
from src.train_regression import train_and_evaluate_regression
from src.train_classification import train_and_evaluate_classification


def test_regression_training_and_metrics():
    """Verify regression models train and yield valid metrics."""
    test_df, cv_df, trained_models = train_and_evaluate_regression(experiment="A")
    assert len(test_df) == 4  # 4 models
    assert set(test_df["Model"].tolist()) == {
        "Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"
    }
    for _, row in test_df.iterrows():
        assert row["MAE"] > 0
        assert row["RMSE"] > 0
        assert -1.0 <= row["R2"] <= 1.0


def test_classification_training_and_metrics():
    """Verify classification models train and yield valid probabilities and scores."""
    test_df, cv_df, trained_models = train_and_evaluate_classification(experiment="B")
    assert len(test_df) == 2  # 2 models
    for _, row in test_df.iterrows():
        assert 0.0 <= row["Accuracy"] <= 1.0
        assert 0.0 <= row["Precision"] <= 1.0
        assert 0.0 <= row["Recall"] <= 1.0
        assert 0.0 <= row["F1"] <= 1.0
        assert 0.0 <= row["ROC_AUC"] <= 1.0


def test_saved_model_artifacts_exist():
    """Check that serialized model files are saved on disk."""
    required_models = [
        "regression_exp_a_random_forest.joblib",
        "regression_exp_b_gradient_boosting.joblib",
        "classification_exp_a_logistic_regression.joblib",
        "classification_exp_b_logistic_regression.joblib",
        "preprocessor_exp_a.joblib",
        "preprocessor_exp_b.joblib"
    ]
    for model_file in required_models:
        path = MODELS_DIR / model_file
        assert path.exists(), f"Missing required model artifact: {path}"
