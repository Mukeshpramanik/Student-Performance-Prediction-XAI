"""
Tests for preprocessing, feature engineering, and train/test separation.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.feature_engineering import add_engineered_features
from src.preprocessing import prepare_data, get_feature_columns


def test_feature_engineering_adds_expected_columns():
    """Verify composite educational indicators are computed accurately."""
    raw_df = load_raw_data()
    clean_df = add_engineered_features(raw_df)

    expected_cols = ["parent_edu_avg", "total_alc", "study_travel_ratio", "academic_support_any", "at_risk"]
    for col in expected_cols:
        assert col in clean_df.columns

    # Verify logic
    assert np.all(clean_df["parent_edu_avg"] == (clean_df["Medu"] + clean_df["Fedu"]) / 2.0)
    assert np.all(clean_df["at_risk"] == (clean_df["G3"] < 10).astype(int))


def test_prepare_data_experiment_a_excludes_g1_g2():
    """Experiment A must strictly omit G1 and G2 to prevent leakage in early prediction."""
    data = prepare_data(experiment="A", target_type="regression")
    assert "G1" not in data["num_cols"]
    assert "G2" not in data["num_cols"]
    assert "G1" not in data["feature_names"]
    assert "G2" not in data["feature_names"]
    assert len(data["X_train"]) + len(data["X_test"]) == 395


def test_prepare_data_experiment_b_includes_g1_g2():
    """Experiment B must include G1 and G2."""
    data = prepare_data(experiment="B", target_type="regression")
    assert "G1" in data["num_cols"]
    assert "G2" in data["num_cols"]
    assert "G1" in data["feature_names"]
    assert "G2" in data["feature_names"]
    assert len(data["feature_names"]) == len(data["X_train"].columns)


def test_classification_stratification():
    """Verify classification split preserves at-risk ratio in train and test sets."""
    data = prepare_data(experiment="A", target_type="classification")
    y_train_ratio = data["y_train"].mean()
    y_test_ratio = data["y_test"].mean()
    assert abs(y_train_ratio - y_test_ratio) < 0.05
