# Student Academic Performance Prediction and Early Intervention System

An Explainable Machine Learning framework for early prediction and analysis of student academic performance using learning analytics and SHAP (SHapley Additive exPlanations).

## Project Overview
This B.Tech CSE Minor Research Project develops a machine learning application and interactive Streamlit dashboard that:
1. Analyzes secondary student performance in mathematics using the UCI Student Performance dataset.
2. Predicts final grades ($G3$: 0–20 scale) using regression models.
3. Classifies academically at-risk students ($G3 < 10$) using classification models.
4. Separates predictions into two realistic timeframes:
   - **Experiment A (Early Prediction)**: Uses only features available before examinations (excludes $G1$ and $G2$).
   - **Experiment B (Mid-Term Prediction)**: Includes interim exam grades ($G1$ and $G2$).
5. Explains predictions with SHAP global beeswarm plots, importance bars, and individual waterfall plots.
6. Delivers rule-based, empathetic intervention recommendations.

## Technology Stack
- **Language**: Python 3
- **Data & Machine Learning**: Pandas, NumPy, Scikit-learn, Joblib
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Application**: Streamlit
- **Testing**: Pytest (16 automated tests)
- **Reporting**: Python-pptx

## Experimental Results Summary

### Regression (Target: G3, 0-20)
- **Experiment A (Early)**: Random Forest achieved MAE = 3.15, RMSE = 3.94, $R^2$ = 0.241 (5-Fold CV MAE: 3.00, $R^2$: 0.192).
- **Experiment B (Mid-Term)**: Gradient Boosting achieved MAE = 1.15, RMSE = 1.95, $R^2$ = 0.814 (5-Fold CV MAE: 1.01, $R^2$: 0.878).

### Classification (Target: At-Risk, G3 < 10)
- **Experiment A (Early)**: Logistic Regression achieved Accuracy = 69.62%, ROC-AUC = 0.711.
- **Experiment B (Mid-Term)**: Logistic Regression achieved Accuracy = 91.14%, Precision = 95.24%, Recall = 76.92%, F1 = 0.851, ROC-AUC = 0.977.

## Directory Structure
- `app.py`: Streamlit 9-page dashboard application.
- `config/settings.py`: Central path and experiment configurations.
- `data/raw/`: Original UCI `student-mat.csv`.
- `data/processed/`: Processed `student_clean.csv`.
- `src/`: Modular Python logic (`data_loader.py`, `preprocessing.py`, `feature_engineering.py`, `eda.py`, `train_regression.py`, `train_classification.py`, `evaluation.py`, `explainability.py`, `predictions.py`, `recommendations.py`).
- `models/`: Serialized Joblib model artifacts and preprocessors.
- `notebooks/`: 4 Jupyter notebooks (data understanding, EDA, training, explainability).
- `reports/figures/`: Publication-quality figures and SHAP charts.
- `reports/tables/`: Statistical tables in CSV format.
- `reports/model_metrics/`: Exported performance metrics for all models.
- `reports/research_paper/minor_research_paper.md`: Full 23-section research paper.
- `reports/Minor_Research_Project_Presentation.pptx`: 20-slide presentation deck (1 title + 19 content slides).
- `docs/`: Project report, architecture diagrams, viva questions (32 questions), and references.
- `tests/`: 16 automated Pytest unit tests.
- `scripts/`: Download script, master training pipeline, presentation generator.

## Quickstart Guide

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Master Training Pipeline
```bash
python scripts/train_all.py
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

## Responsible AI & Ethics
This application is designed as a decision-support aid for educators and academic advisors. Predictions are non-binding estimates and must never be used to automate punitive academic decisions.
