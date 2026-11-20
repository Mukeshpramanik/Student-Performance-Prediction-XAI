"""
Explainable AI (XAI) module powered by SHAP for global and local interpretability.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib.pyplot as plt
import shap

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import MODELS_DIR, FIGURES_DIR
from src.preprocessing import prepare_data


def get_explainer(model: Any, X_background: Optional[pd.DataFrame] = None) -> Any:
    """Select the appropriate SHAP explainer based on model architecture."""
    model_type = type(model).__name__.lower()

    if any(tree_name in model_type for tree_name in ["forest", "tree", "gradientboosting"]):
        return shap.TreeExplainer(model)
    elif "linear" in model_type or "logistic" in model_type:
        if X_background is not None:
            # Use background sample for LinearExplainer
            background_sample = shap.sample(X_background, min(50, len(X_background)), random_state=42)
            return shap.LinearExplainer(model, background_sample)
        return shap.Explainer(model)
    else:
        if X_background is not None:
            background_sample = shap.sample(X_background, min(50, len(X_background)), random_state=42)
            return shap.Explainer(model, background_sample)
        return shap.Explainer(model)


def compute_shap_values(model: Any, X_data: pd.DataFrame, X_background: Optional[pd.DataFrame] = None) -> shap.Explanation:
    """Compute SHAP values for the provided dataset."""
    explainer = get_explainer(model, X_background)
    shap_vals = explainer(X_data)
    return shap_vals


def plot_summary(
    shap_values: shap.Explanation,
    X_data: pd.DataFrame,
    max_display: int = 15,
    save_path: Optional[Path] = None,
    title: str = "SHAP Beeswarm Summary Plot"
) -> plt.Figure:
    """Generate and optionally save a SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_global_bar(
    shap_values: shap.Explanation,
    max_display: int = 15,
    save_path: Optional[Path] = None,
    title: str = "Mean Absolute SHAP Value (Global Importance)"
) -> plt.Figure:
    """Generate and optionally save a global SHAP bar plot."""
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_local_waterfall(
    shap_single: shap.Explanation,
    max_display: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Individual Prediction Contribution (SHAP Waterfall)"
) -> plt.Figure:
    """Generate a waterfall plot illustrating feature attribution for a single student."""
    plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(shap_single, max_display=max_display, show=False)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def get_feature_impact_table(shap_single: shap.Explanation, top_n: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the top positive (grade-increasing) and negative (grade-decreasing) feature attributions.
    """
    feature_names = shap_single.feature_names
    values = shap_single.values
    data_vals = shap_single.data

    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "Observed_Value": [round(float(v), 2) if isinstance(v, (int, float, np.number)) else str(v) for v in data_vals],
        "SHAP_Impact": values
    })

    top_positive = df_contrib[df_contrib["SHAP_Impact"] > 0].sort_values(by="SHAP_Impact", ascending=False).head(top_n)
    top_negative = df_contrib[df_contrib["SHAP_Impact"] < 0].sort_values(by="SHAP_Impact", ascending=True).head(top_n)

    return top_positive, top_negative


def generate_all_shap_reports(output_dir: Path = FIGURES_DIR) -> Dict[str, Path]:
    """Compute and persist publication-ready SHAP plots for Experiments A and B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # Experiment A: Early Prediction (Random Forest Regressor)
    data_a = prepare_data(experiment="A")
    model_a = joblib.load(MODELS_DIR / "regression_exp_a_random_forest.joblib")
    shap_a = compute_shap_values(model_a, data_a["X_test"], data_a["X_train"])

    p_summary_a = output_dir / "07_shap_summary_exp_a.png"
    plot_summary(shap_a, data_a["X_test"], save_path=p_summary_a, title="SHAP Summary Plot - Early Prediction (Exp A)")
    plt.close()
    generated["shap_summary_exp_a"] = p_summary_a

    p_bar_a = output_dir / "08_shap_bar_exp_a.png"
    plot_global_bar(shap_a, save_path=p_bar_a, title="Global Feature Importance - Early Prediction (Exp A)")
    plt.close()
    generated["shap_bar_exp_a"] = p_bar_a

    # Experiment B: Mid-Term Prediction (Gradient Boosting Regressor)
    data_b = prepare_data(experiment="B")
    model_b = joblib.load(MODELS_DIR / "regression_exp_b_gradient_boosting.joblib")
    shap_b = compute_shap_values(model_b, data_b["X_test"], data_b["X_train"])

    p_summary_b = output_dir / "09_shap_summary_exp_b.png"
    plot_summary(shap_b, data_b["X_test"], save_path=p_summary_b, title="SHAP Summary Plot - Mid-Term Prediction (Exp B)")
    plt.close()
    generated["shap_summary_exp_b"] = p_summary_b

    p_bar_b = output_dir / "10_shap_bar_exp_b.png"
    plot_global_bar(shap_b, save_path=p_bar_b, title="Global Feature Importance - Mid-Term Prediction (Exp B)")
    plt.close()
    generated["shap_bar_exp_b"] = p_bar_b

    return generated


if __name__ == "__main__":
    report_paths = generate_all_shap_reports()
    print("Generated SHAP visual reports:")
    for k, v in report_paths.items():
        print(f"  {k}: {v}")
