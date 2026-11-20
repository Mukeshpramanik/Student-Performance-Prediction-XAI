"""
Evaluation metrics computation, cross-validation, and subgroup performance analysis.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate


def evaluate_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 score for regression models."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1, and ROC-AUC for classification models."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1)
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_proba)
            metrics["ROC_AUC"] = float(auc)
        except Exception:
            metrics["ROC_AUC"] = 0.0
    else:
        metrics["ROC_AUC"] = 0.0

    return metrics


def evaluate_subgroup_fairness(
    df_raw_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str = "sex",
    task: str = "regression"
) -> pd.DataFrame:
    """
    Subgroup performance evaluation across sensitive/demographic attributes.
    Reports sample counts, subgroup error metrics, and statistical differences.
    """
    df_eval = df_raw_test.copy().reset_index(drop=True)
    df_eval["y_true"] = np.array(y_true)
    df_eval["y_pred"] = np.array(y_pred)

    records = []
    for grp_val, grp_data in df_eval.groupby(group_col):
        n_samples = len(grp_data)
        if task == "regression":
            mae = mean_absolute_error(grp_data["y_true"], grp_data["y_pred"])
            rmse = root_mean_squared_error(grp_data["y_true"], grp_data["y_pred"])
            r2 = r2_score(grp_data["y_true"], grp_data["y_pred"]) if n_samples > 2 else np.nan
            records.append({
                "Group_Attribute": group_col,
                "Group_Value": grp_val,
                "Sample_Size": n_samples,
                "Sample_Ratio": n_samples / len(df_eval),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
        else:
            acc = accuracy_score(grp_data["y_true"], grp_data["y_pred"])
            prec = precision_score(grp_data["y_true"], grp_data["y_pred"], zero_division=0)
            rec = recall_score(grp_data["y_true"], grp_data["y_pred"], zero_division=0)
            f1 = f1_score(grp_data["y_true"], grp_data["y_pred"], zero_division=0)
            records.append({
                "Group_Attribute": group_col,
                "Group_Value": grp_val,
                "Sample_Size": n_samples,
                "Sample_Ratio": n_samples / len(df_eval),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            })

    return pd.DataFrame(records)
