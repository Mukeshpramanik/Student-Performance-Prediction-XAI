"""
Tests for inference pipelines, single student predictions, batch CSV validation, and XAI attribution.
"""

import pytest
import pandas as pd
from src.predictions import predict_single_student, predict_batch_csv, generate_sample_template_csv


@pytest.fixture
def sample_student_record():
    return {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 4, "Fedu": 3, "Mjob": "services", "Fjob": "teacher", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no",
        "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3, "goout": 2,
        "Dalc": 1, "Walc": 1, "health": 4, "absences": 4
    }


def test_predict_single_student_experiment_a(sample_student_record):
    """Test single inference in Experiment A mode."""
    res = predict_single_student(sample_student_record, experiment="A")
    assert "predicted_grade" in res
    assert 0.0 <= res["predicted_grade"] <= 20.0
    assert res["ci_lower"] <= res["predicted_grade"] <= res["ci_upper"]
    assert 0.0 <= res["at_risk_probability"] <= 1.0
    assert isinstance(res["is_at_risk"], bool)
    assert len(res["recommendations"]) > 0


def test_predict_single_student_experiment_b(sample_student_record):
    """Test single inference in Experiment B mode."""
    record_b = sample_student_record.copy()
    record_b["G1"] = 15
    record_b["G2"] = 16
    res = predict_single_student(record_b, experiment="B")
    assert 0.0 <= res["predicted_grade"] <= 20.0
    assert res["predicted_grade"] > 10.0  # High G1/G2 student should score above pass mark


def test_batch_csv_prediction_success():
    """Verify batch CSV prediction successfully parses valid schema."""
    csv_bytes = generate_sample_template_csv("A").encode("utf-8")
    results_df, summary = predict_batch_csv(csv_bytes, experiment="A")
    assert summary["total_records"] == 2
    assert "Predicted_G3" in results_df.columns
    assert "At_Risk_Classification" in results_df.columns
    assert "Risk_Category" in results_df.columns


def test_batch_csv_prediction_invalid_columns():
    """Verify batch prediction raises informative error on missing columns."""
    bad_csv = b"col1,col2\n1,2"
    with pytest.raises(ValueError, match="missing mandatory columns"):
        predict_batch_csv(bad_csv, experiment="A")
