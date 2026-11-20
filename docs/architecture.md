# System Architecture & Component Interaction

## Architecture Overview
The system follows a decoupled, three-tier architecture:
1. **Presentation & Application Layer**: Interactive Streamlit multi-page frontend.
2. **Business Logic & Inference Layer**: Scikit-Learn pipelines, SHAP explainers, and rule-based recommendation engine.
3. **Data & Persistence Layer**: File-based storage holding raw CSVs, processed datasets, and serialized Joblib model artifacts.

---

## Architecture Flow Diagram

```mermaid
flowchart TD
    subgraph Data Layer
        A1["Raw Data: student-mat.csv"] --> A2["Data Loader & Validator"]
        A2 --> A3["Feature Engineering Engine"]
        A3 --> A4["Processed Data: student_clean.csv"]
    end

    subgraph Preprocessing & Model Training
        A4 --> B1["Leakage-Free Train/Test Split (80/20)"]
        B1 --> B2["ColumnTransformer (Scaler + OneHotEncoder)"]
        B2 --> B3["Experiment A: Early Models (No G1/G2)"]
        B2 --> B4["Experiment B: Mid-Term Models (With G1/G2)"]
        B3 --> B5["Joblib Serialization: models/"]
        B4 --> B5
    end

    subgraph Inference & Explainability
        B5 --> C1["Prediction Service (src/predictions.py)"]
        B5 --> C2["SHAP Engine (src/explainability.py)"]
        C1 --> C3["Uncertainty Estimator (Tree Variance Spread)"]
        C1 --> C4["Rule-Based Recommendation Engine"]
    end

    subgraph UI & Presentation
        C1 --> D1["Streamlit Multi-Page App (app.py)"]
        C2 --> D1
        C3 --> D1
        C4 --> D1
    end
```

---

## Component Breakdown
- `config/settings.py`: Centralized directory paths, random seed (42), and target thresholds.
- `src/data_loader.py`: Semicolon-delimited CSV reader, schema assertion, null/duplicate checker.
- `src/feature_engineering.py`: Computes derived educational features and binary at-risk targets.
- `src/preprocessing.py`: Implements `ColumnTransformer` with strict separation between Experiments A and B.
- `src/train_regression.py` & `src/train_classification.py`: Executes 5-fold cross-validation, fits models, and exports joblib artifacts.
- `src/explainability.py`: Manages SHAP TreeExplainer and LinearExplainer pipelines, generating summary and waterfall plots.
- `src/predictions.py`: Handles individual and batch CSV predictions, uncertainty bounds, and schema validation.
- `src/recommendations.py`: Multi-factor rule-based logic generating empathetic, non-judgmental remediation advice.
- `app.py`: Modern, responsive 9-page dashboard built with Streamlit and Plotly.
