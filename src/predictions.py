"""
Inference pipeline for single-student predictions and batch CSV evaluation with uncertainty estimation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import io
import joblib
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import MODELS_DIR, SEED, AT_RISK_THRESHOLD
from src.feature_engineering import add_engineered_features
from src.preprocessing import get_feature_columns, get_feature_names_from_transformer
from src.explainability import compute_shap_values, get_feature_impact_table
from src.recommendations import generate_recommendations


def load_artifacts(experiment: str = "A") -> Dict[str, Any]:
    """Load preprocessor, best regression, and best classification model for given experiment."""
    exp_suffix = experiment.lower()

    preprocessor_path = MODELS_DIR / f"preprocessor_exp_{exp_suffix}.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor for Experiment {experiment}. Run training first.")

    preprocessor = joblib.load(preprocessor_path)

    # Best regression models per experiment
    if exp_suffix == "a":
        reg_model_path = MODELS_DIR / "regression_exp_a_random_forest.joblib"
        clf_model_path = MODELS_DIR / "classification_exp_a_logistic_regression.joblib"
    else:
        reg_model_path = MODELS_DIR / "regression_exp_b_gradient_boosting.joblib"
        clf_model_path = MODELS_DIR / "classification_exp_b_logistic_regression.joblib"

    reg_model = joblib.load(reg_model_path)
    clf_model = joblib.load(clf_model_path)

    return {
        "preprocessor": preprocessor,
        "regression_model": reg_model,
        "classification_model": clf_model,
        "experiment": experiment
    }


def predict_single_student(
    student_record: Dict[str, Any],
    experiment: str = "A",
    artifacts: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate predicted final grade, confidence intervals, at-risk classification, and SHAP explanation.
    """
    if artifacts is None:
        artifacts = load_artifacts(experiment)

    preprocessor = artifacts["preprocessor"]
    reg_model = artifacts["regression_model"]
    clf_model = artifacts["classification_model"]

    # Convert single record to DataFrame
    raw_df = pd.DataFrame([student_record])
    feat_df = add_engineered_features(raw_df)

    cat_cols, num_cols = get_feature_columns(experiment)
    req_cols = cat_cols + num_cols

    for col in req_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X_subset = feat_df[req_cols]
    X_trans = preprocessor.transform(X_subset)
    feature_names = get_feature_names_from_transformer(preprocessor, cat_cols, num_cols)
    X_df = pd.DataFrame(X_trans, columns=feature_names)

    # 1. Regression Prediction (G3)
    pred_g3 = float(reg_model.predict(X_df)[0])
    pred_g3_clamped = max(0.0, min(20.0, pred_g3))

    # Uncertainty Estimation: Use ensemble tree variance if available
    X_mat = X_df.values
    if hasattr(reg_model, "estimators_"):
        try:
            estimators_list = reg_model.estimators_.flatten() if hasattr(reg_model.estimators_, "flatten") else reg_model.estimators_
            tree_preds = [
                float(est.predict(X_mat)[0]) if hasattr(est, "predict") else float(est[0].predict(X_mat)[0])
                for est in estimators_list
            ]
            std_err = float(np.std(tree_preds))
        except Exception:
            std_err = 1.2
        ci_lower = max(0.0, pred_g3_clamped - 1.96 * std_err)
        ci_upper = min(20.0, pred_g3_clamped + 1.96 * std_err)
    else:
        std_err = 1.5
        ci_lower = max(0.0, pred_g3_clamped - 1.96 * std_err)
        ci_upper = min(20.0, pred_g3_clamped + 1.96 * std_err)

    # 2. Classification Prediction (At-Risk: G3 < 10)
    if hasattr(clf_model, "predict_proba"):
        at_risk_proba = float(clf_model.predict_proba(X_df)[0, 1])
    else:
        at_risk_proba = 1.0 if pred_g3_clamped < AT_RISK_THRESHOLD else 0.0

    is_at_risk = bool(at_risk_proba >= 0.5)

    if at_risk_proba >= 0.65:
        risk_level = "High Risk"
    elif at_risk_proba >= 0.35:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # 3. SHAP Explanation
    shap_explanation = compute_shap_values(reg_model, X_df)
    single_shap = shap_explanation[0]
    top_pos, top_neg = get_feature_impact_table(single_shap, top_n=5)

    # 4. Supportive Recommendations
    recommendations = generate_recommendations(student_record, pred_g3_clamped, at_risk_proba)

    return {
        "predicted_grade": round(pred_g3_clamped, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "prediction_uncertainty_std": round(std_err, 2),
        "at_risk_probability": round(at_risk_proba, 3),
        "is_at_risk": is_at_risk,
        "risk_level": risk_level,
        "shap_explanation": single_shap,
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "recommendations": recommendations,
        "transformed_features": X_df
    }


def predict_batch_csv(
    file_content: bytes,
    experiment: str = "A"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process an uploaded CSV dataset and return predicted grades, at-risk classifications, and summary metrics.
    """
    try:
        df_raw = pd.read_csv(io.BytesIO(file_content), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Unable to parse CSV file: {str(e)}")

    artifacts = load_artifacts(experiment)
    preprocessor = artifacts["preprocessor"]
    reg_model = artifacts["regression_model"]
    clf_model = artifacts["classification_model"]

    df_feat = add_engineered_features(df_raw)
    cat_cols, num_cols = get_feature_columns(experiment)
    req_cols = cat_cols + num_cols

    # Verify column presence
    missing_cols = [c for c in req_cols if c not in df_feat.columns]
    if missing_cols:
        raise ValueError(f"Uploaded CSV is missing mandatory columns for Experiment {experiment}: {missing_cols[:5]}...")

    X_subset = df_feat[req_cols]
    X_trans = preprocessor.transform(X_subset)
    feature_names = get_feature_names_from_transformer(preprocessor, cat_cols, num_cols)
    X_df = pd.DataFrame(X_trans, columns=feature_names)

    # Regression Predictions
    g3_preds = reg_model.predict(X_df)
    g3_preds_clamped = np.clip(g3_preds, 0.0, 20.0)

    # Classification Predictions
    if hasattr(clf_model, "predict_proba"):
        at_risk_probas = clf_model.predict_proba(X_df)[:, 1]
    else:
        at_risk_probas = (g3_preds_clamped < AT_RISK_THRESHOLD).astype(float)

    results_df = df_raw.copy()
    results_df["Predicted_G3"] = np.round(g3_preds_clamped, 2)
    results_df["At_Risk_Probability"] = np.round(at_risk_probas, 3)
    results_df["At_Risk_Classification"] = np.where(at_risk_probas >= 0.5, "At Risk (<10)", "Not At Risk (>=10)")
    results_df["Risk_Category"] = np.where(
        at_risk_probas >= 0.65, "High Risk",
        np.where(at_risk_probas >= 0.35, "Moderate Risk", "Low Risk")
    )

    summary = {
        "total_records": len(results_df),
        "mean_predicted_g3": float(np.mean(g3_preds_clamped)),
        "at_risk_count": int(np.sum(at_risk_probas >= 0.5)),
        "at_risk_rate": float(np.mean(at_risk_probas >= 0.5)),
        "high_risk_count": int(np.sum(at_risk_probas >= 0.65))
    }

    return results_df, summary


def generate_sample_template_csv(experiment: str = "A") -> str:
    """Produce a template CSV text suitable for batch upload."""
    sample_data = {
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "age": [17, 16],
        "address": ["U", "R"],
        "famsize": ["GT3", "LE3"],
        "Pstatus": ["T", "A"],
        "Medu": [3, 2],
        "Fedu": [3, 1],
        "Mjob": ["services", "other"],
        "Fjob": ["teacher", "services"],
        "reason": ["course", "reputation"],
        "guardian": ["mother", "father"],
        "traveltime": [1, 2],
        "studytime": [2, 1],
        "failures": [0, 1],
        "schoolsup": ["no", "yes"],
        "famsup": ["yes", "no"],
        "paid": ["no", "yes"],
        "activities": ["yes", "no"],
        "nursery": ["yes", "yes"],
        "higher": ["yes", "yes"],
        "internet": ["yes", "no"],
        "romantic": ["no", "yes"],
        "famrel": [4, 3],
        "freetime": [3, 2],
        "goout": [3, 4],
        "Dalc": [1, 2],
        "Walc": [1, 3],
        "health": [4, 3],
        "absences": [4, 12]
    }
    if experiment.upper() == "B":
        sample_data["G1"] = [12, 7]
        sample_data["G2"] = [11, 8]

    df_sample = pd.DataFrame(sample_data)
    return df_sample.to_csv(index=False)
