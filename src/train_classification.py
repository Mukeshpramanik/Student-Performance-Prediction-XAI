"""
Classification model training, stratified cross-validation, and artifact persistence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import MODELS_DIR, METRICS_DIR, SEED
from src.preprocessing import prepare_data
from src.evaluation import evaluate_classification_predictions


def get_classification_models(seed: int = SEED) -> Dict[str, Any]:
    """Instantiate the 2 designated classification models with fixed random seeds."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    }


def train_and_evaluate_classification(experiment: str = "A") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train classification models for the specified experiment (A: Early, B: Mid-Term).
    Evaluates 5-fold stratified cross-validation on train set and evaluation on held-out test set.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_data(experiment=experiment, target_type="classification", random_state=SEED)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    models = get_classification_models(seed=SEED)
    test_results = []
    cv_results = []
    trained_models = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        # Stratified 5-Fold Cross Validation
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
        cv_acc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        cv_roc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")

        cv_results.append({
            "Experiment": f"Experiment {experiment}",
            "Model": name,
            "CV_Accuracy_Mean": float(np.mean(cv_acc_scores)),
            "CV_Accuracy_Std": float(np.std(cv_acc_scores)),
            "CV_F1_Mean": float(np.mean(cv_f1_scores)),
            "CV_F1_Std": float(np.std(cv_f1_scores)),
            "CV_ROC_AUC_Mean": float(np.mean(cv_roc_scores)),
            "CV_ROC_AUC_Std": float(np.std(cv_roc_scores))
        })

        # Fit on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_classification_predictions(y_test, y_pred, y_proba)

        test_results.append({
            "Experiment": f"Experiment {experiment}",
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "ROC_AUC": metrics["ROC_AUC"]
        })

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"classification_exp_{experiment.lower()}_{safe_name}.joblib"
        joblib.dump(model, model_path)
        trained_models[name] = model

    test_df = pd.DataFrame(test_results)
    cv_df = pd.DataFrame(cv_results)

    return test_df, cv_df, trained_models


def run_all_classification_experiments() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute classification training across both Experiment A and Experiment B."""
    test_a, cv_a, _ = train_and_evaluate_classification(experiment="A")
    test_b, cv_b, _ = train_and_evaluate_classification(experiment="B")

    combined_test = pd.concat([test_a, test_b], ignore_index=True)
    combined_cv = pd.concat([cv_a, cv_b], ignore_index=True)

    combined_test.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)
    combined_cv.to_csv(METRICS_DIR / "cv_classification_metrics.csv", index=False)

    return combined_test, combined_cv


if __name__ == "__main__":
    test_metrics, cv_metrics = run_all_classification_experiments()
    print("=== Classification Test Set Metrics ===")
    print(test_metrics.to_string(index=False))
    print("\n=== Classification Stratified 5-Fold Cross Validation Metrics ===")
    print(cv_metrics.to_string(index=False))
