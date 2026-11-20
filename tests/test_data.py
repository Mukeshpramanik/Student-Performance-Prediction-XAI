"""
Tests for data loading, integrity verification, and dataset validation.
"""

import pytest
import pandas as pd
from pathlib import Path

from config.settings import RAW_DATA_FILE, PROCESSED_DATA_FILE
from src.data_loader import load_raw_data, validate_dataset, save_processed_data, load_processed_data


def test_raw_dataset_exists_and_loads():
    """Verify raw dataset exists and loads with correct dimensions."""
    assert RAW_DATA_FILE.exists(), f"Raw dataset not found at {RAW_DATA_FILE}"
    df = load_raw_data(RAW_DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 395
    assert df.shape[1] == 33


def test_dataset_integrity():
    """Ensure raw dataset has zero missing values and zero duplicate rows."""
    df = load_raw_data(RAW_DATA_FILE)
    val = validate_dataset(df)
    assert val["missing_values_total"] == 0
    assert val["duplicate_rows"] == 0
    assert val["target_present"] is True
    assert 0 <= val["target_min"] <= 20
    assert 0 <= val["target_max"] <= 20


def test_processed_data_save_and_load(tmp_path):
    """Test saving and loading processed datasets."""
    dummy_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    temp_file = tmp_path / "test_proc.csv"
    save_processed_data(dummy_df, temp_file)
    loaded = load_processed_data(temp_file)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["a", "b"]
