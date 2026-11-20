"""
Master training and artifact generation pipeline.
Executes data loading, model training, cross-validation, and XAI report generation.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data, save_processed_data, validate_dataset
from src.feature_engineering import add_engineered_features
from src.eda import run_full_eda
from src.train_regression import run_all_regression_experiments
from src.train_classification import run_all_classification_experiments
from src.explainability import generate_all_shap_reports
from src.evaluation import evaluate_subgroup_fairness
from src.preprocessing import prepare_data
import joblib
from config.settings import METRICS_DIR, MODELS_DIR


def run_pipeline():
    print("=" * 60)
    print("STARTING END-TO-END TRAINING AND EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load and process data
    print("\n[Step 1/6] Loading and validating raw dataset...")
    raw_df = load_raw_data()
    val_info = validate_dataset(raw_df)
    print(f"  Records: {val_info['num_rows']} | Columns: {val_info['num_columns']} | Nulls: {val_info['missing_values_total']}")
    clean_df = add_engineered_features(raw_df)
    save_processed_data(clean_df)
    print("  Processed dataset saved to data/processed/student_clean.csv")

    # 2. EDA Generation
    print("\n[Step 2/6] Generating EDA figures and tables...")
    eda_res = run_full_eda()
    print(f"  Generated {len(eda_res['figures'])} figures and {len(eda_res['tables'])} statistical tables.")

    # 3. Regression Models
    print("\n[Step 3/6] Training regression models (Experiments A & B)...")
    reg_test, reg_cv = run_all_regression_experiments()
    print("  Regression test metrics logged to reports/model_metrics/regression_metrics.csv")

    # 4. Classification Models
    print("\n[Step 4/6] Training classification models (Experiments A & B)...")
    clf_test, clf_cv = run_all_classification_experiments()
    print("  Classification test metrics logged to reports/model_metrics/classification_metrics.csv")

    # 5. SHAP Explainability
    print("\n[Step 5/6] Generating global and local SHAP explanations...")
    shap_reports = generate_all_shap_reports()
    print(f"  Generated {len(shap_reports)} publication-grade SHAP plots.")

    # 6. Subgroup Fairness Evaluation
    print("\n[Step 6/6] Computing subgroup fairness evaluations...")
    data_b = prepare_data(experiment="B", target_type="regression")
    model_b = joblib.load(MODELS_DIR / "regression_exp_b_gradient_boosting.joblib")
    y_pred_b = model_b.predict(data_b["X_test"])
    fairness_df = evaluate_subgroup_fairness(data_b["X_test_raw"], data_b["y_test"].values, y_pred_b, group_col="sex")
    fairness_df.to_csv(METRICS_DIR / "fairness_subgroup_sex.csv", index=False)
    print("  Fairness subgroup analysis saved to reports/model_metrics/fairness_subgroup_sex.csv")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
