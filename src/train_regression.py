"""
Regression model training, cross-validation, and artifact persistence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import MODELS_DIR, METRICS_DIR, SEED
from src.preprocessing import prepare_data
from src.evaluation import evaluate_regression_predictions


def get_regression_models(seed: int = SEED) -> Dict[str, Any]:
    """Instantiate the 4 designated regression models with fixed random seeds."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=seed, max_depth=5),
        "Random Forest": RandomForestRegressor(random_state=seed, n_estimators=100, max_depth=6),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed, n_estimators=100, learning_rate=0.05, max_depth=3)
    }


def train_and_evaluate_regression(experiment: str = "A") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train regression models for the specified experiment (A: Early, B: Mid-Term).
    Computes 5-fold cross-validation on train set and evaluation on held-out test set.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_data(experiment=experiment, target_type="regression", random_state=SEED)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]

    # Save preprocessor
    joblib.dump(preprocessor, MODELS_DIR / f"preprocessor_exp_{experiment.lower()}.joblib")

    models = get_regression_models(seed=SEED)
    test_results = []
    cv_results = []
    trained_models = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        # 5-Fold Cross Validation on Training Data (negative MSE / MAE / R2)
        cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

        cv_results.append({
            "Experiment": f"Experiment {experiment}",
            "Model": name,
            "CV_MAE_Mean": float(np.mean(cv_mae_scores)),
            "CV_MAE_Std": float(np.std(cv_mae_scores)),
            "CV_R2_Mean": float(np.mean(cv_r2_scores)),
            "CV_R2_Std": float(np.std(cv_r2_scores))
        })

        # Fit on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression_predictions(y_test, y_pred)

        test_results.append({
            "Experiment": f"Experiment {experiment}",
            "Model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"regression_exp_{experiment.lower()}_{safe_name}.joblib"
        joblib.dump(model, model_path)
        trained_models[name] = model

    test_df = pd.DataFrame(test_results)
    cv_df = pd.DataFrame(cv_results)

    return test_df, cv_df, trained_models


def run_all_regression_experiments() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute regression training across both Experiment A and Experiment B."""
    test_a, cv_a, _ = train_and_evaluate_regression(experiment="A")
    test_b, cv_b, _ = train_and_evaluate_regression(experiment="B")

    combined_test = pd.concat([test_a, test_b], ignore_index=True)
    combined_cv = pd.concat([cv_a, cv_b], ignore_index=True)

    combined_test.to_csv(METRICS_DIR / "regression_metrics.csv", index=False)
    combined_cv.to_csv(METRICS_DIR / "cv_regression_metrics.csv", index=False)

    return combined_test, combined_cv


if __name__ == "__main__":
    test_metrics, cv_metrics = run_all_regression_experiments()
    print("=== Regression Test Set Metrics ===")
    print(test_metrics.to_string(index=False))
    print("\n=== Regression 5-Fold Cross Validation Metrics ===")
    print(cv_metrics.to_string(index=False))
