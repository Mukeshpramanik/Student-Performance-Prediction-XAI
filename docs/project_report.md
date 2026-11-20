# B.Tech Minor Project Report
## Explainable AI-Based Student Academic Performance Prediction and Early Intervention System

---

### 1. Cover Page
**Project Title:** Explainable AI-Based Student Academic Performance Prediction and Early Intervention System  
**Academic Degree:** Bachelor of Technology in Computer Science & Engineering  
**Student Name:** Mukesh Pramanik  
**Repository:** https://github.com/Mukeshpramanik/Student-Performance-Prediction-XAI  
**Academic Year:** 2026-2027  

---

### 2. Certificate Placeholder
```
                         CERTIFICATE
This is to certify that the Minor Project Report entitled:
"Explainable AI-Based Student Academic Performance Prediction and Early Intervention System"
submitted by:
                     Mukesh Pramanik
in partial fulfillment of the requirements for the award of the degree of
Bachelor of Technology in Computer Science and Engineering is an authentic
record of the research and development work carried out under supervision.

Internal Supervisor: _______________________      Date: __________________
Head of Department:  _______________________      Date: __________________
```

---

### 3. Declaration Placeholder
```
                         DECLARATION
I, Mukesh Pramanik, hereby declare that this project report entitled
"Explainable AI-Based Student Academic Performance Prediction and Early Intervention System"
is an authentic record of my own research and engineering work. No portion of this work
has been submitted elsewhere for the award of any degree or diploma.

Signature: _______________________                Date: __________________
Name: Mukesh Pramanik
```

---

### 4. Acknowledgement
I express my deepest gratitude to my academic supervisors, faculty mentors, and peer reviewers in the Department of Computer Science & Engineering for their valuable guidance, insightful feedback, and continuous encouragement throughout the design and execution of this minor research project. I also thank the UC Irvine Center for Machine Learning and Intelligent Systems for providing access to the benchmark educational dataset.

---

### 5. Abstract
Early identification of academically vulnerable students is vital for administering proactive pedagogical support. However, standard machine learning architectures function as opaque black boxes, preventing practical institutional adoption. This project designs, develops, tests, and deploys an Explainable Artificial Intelligence (XAI) framework for student performance prediction and early intervention. Utilizing the UCI Student Performance dataset ($N=395$), we implement two rigorous experiments: **Experiment A (Early Prediction)** using solely pre-examination background factors, and **Experiment B (Mid-Term Prediction)** incorporating interim evaluation marks ($G1, G2$). In Experiment A, Random Forest achieves an MAE of 3.15 and $R^2$ of 0.241, while Logistic Regression scores an At-Risk ROC-AUC of 0.711. In Experiment B, Gradient Boosting achieves an MAE of 1.15, RMSE of 1.95, and $R^2$ of 0.814, with Logistic Regression reaching 91.14% accuracy and 0.977 ROC-AUC. SHAP explainability provides global cohort feature importance and localized individual waterfall plots. The entire framework is deployed via an interactive 9-page Streamlit dashboard featuring automated, supportive pedagogical guidance.

---

### 6. Introduction
Educational institutions face significant challenges regarding student academic attrition and performance variation. While traditional academic reviews are retrospective, modern learning analytics offers predictive capabilities. This project bridges predictive machine learning and human pedagogical judgment by pairing high-performance ensemble models with game-theoretic explainability.

---

### 7. Problem Statement
Current academic monitoring systems suffer from four critical deficiencies:
1. **Reactive Assessment**: Flagging student distress only after exams conclude.
2. **Opacity of Predictions**: Inability to justify why a specific student is labeled at risk.
3. **Temporal Leakage**: Conflating early pre-exam factors with mid-term grades.
4. **Lack of Actionable Advice**: Outputting isolated scores without guidance for intervention.

---

### 8. Objectives
- Implement and benchmark regression and classification algorithms for academic performance prediction.
- Structure separate early (pre-exam) and mid-term predictive pipelines to prevent temporal leakage.
- Integrate SHAP explainability at global and individual student levels.
- Evaluate subgroup fairness across demographic cohorts (biological sex).
- Build a responsive, 9-page Streamlit decision-support application.

---

### 9. Literature Review
- **Cortez & Silva (2008)**: Established the benchmark secondary school dataset, finding strong predictive power in past failures and interim evaluations.
- **Lundberg & Lee (2017)**: Formulated SHAP, establishing the mathematical foundations for local feature attribution in complex machine learning models.
- **Romero & Ventura (2020)**: Highlighted that teacher trust and interpretability are paramount for the real-world success of educational analytics.
- **Conati et al. (2018)**: Emphasized that AI in education requires interpretable reasoning to support teacher-led decision making.

---

### 10. System Requirements
- **Hardware**: Compatible with local execution on MacBook Air (Apple Silicon or Intel), minimum 8GB RAM.
- **Operating System**: macOS / Linux / Windows.
- **Software & Runtimes**: Python 3.10 to 3.14.
- **Key Libraries**: NumPy, Pandas, Scikit-learn, SHAP, Streamlit, Plotly, Matplotlib, Seaborn, Pytest, Python-pptx.

---

### 11. System Architecture
```
[UCI Raw Dataset (CSV)] 
          │
          ▼
[Data Ingestion & Integrity Validation] 
          │
          ▼
[Feature Engineering] (parent_edu_avg, total_alc, study_travel_ratio, at_risk)
          │
          ▼
[ColumnTransformer Preprocessing] ── (StandardScaler + OneHotEncoder)
          │
    ┌─────┴────────────────────────┐
    ▼                              ▼
[Experiment A: Early Pipeline]  [Experiment B: Mid-Term Pipeline]
(Excludes G1, G2)              (Includes G1, G2)
    │                              │
    ▼                              ▼
[Model Training & 5-Fold CV]    [Model Training & 5-Fold CV]
(Linear, Tree, RF, GBDT)       (Linear, Tree, RF, GBDT)
    │                              │
    └──────────────┬───────────────┘
                   ▼
       [Model Evaluation & Joblib Storage]
                   │
    ┌──────────────┴───────────────┐
    ▼                              ▼
[SHAP Explainability Engine]    [Rule-Based Support Engine]
(Summary, Bar, Waterfall)      (Attendance, Study, Remediation)
    │                              │
    └──────────────┬───────────────┘
                   ▼
  [Streamlit 9-Page Interactive Dashboard]
```

---

### 12. Methodology
- **Data Split**: 80% train, 20% test split with fixed seed (42).
- **Leakage Prevention**: Preprocessors fitted strictly on training data.
- **Model Suite**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression.
- **Evaluation**: 5-Fold Cross Validation alongside held-out test evaluation.

---

### 13. Dataset Description
- **Source**: UCI Student Performance (Mathematics course: `student-mat.csv`).
- **Instances**: 395 students.
- **Attributes**: 33 features covering demographic, parental, academic, behavioral, and examination scores.
- **Integrity**: 0 missing values, 0 duplicates.

---

### 14. Implementation
The codebase is structured into modular Python components:
- `src/data_loader.py`: Ingestion and schema validation.
- `src/feature_engineering.py`: Creation of composite educational indicators.
- `src/preprocessing.py`: Separation of Experiment A/B pipelines.
- `src/train_regression.py` & `src/train_classification.py`: Model fitting, CV, and serialization.
- `src/explainability.py`: TreeExplainer and LinearExplainer calculations.
- `src/predictions.py`: Single inference with tree-ensemble spread estimates and batch CSV parsing.
- `src/recommendations.py`: Empathetic intervention guidance.
- `app.py`: Streamlit frontend with Plotly visualizations.

---

### 15. Results and Discussion
- **Regression**: Gradient Boosting achieved $R^2 = 0.814$ and MAE = 1.15 in Experiment B, outperforming all baseline models. In Experiment A, Random Forest proved most resilient ($R^2 = 0.241$, MAE = 3.15).
- **Classification**: Logistic Regression achieved 91.14% accuracy and 0.977 ROC-AUC in Experiment B, and 0.711 ROC-AUC in Experiment A.
- **Explainable AI (SHAP)**:
  - Feature importance is measured as mean absolute SHAP values ($\text{mean}(|\text{SHAP value}|)$), reflecting observational attribution rather than percentage contributions or causal effects.
  - *Experiment A (Early)*: `absences` (0.9813), `failures` (0.9097), `health` (0.1606), `goout` (0.1485), `parent_edu_avg` (0.1403).
  - *Experiment B (Mid-Term)*: `G2` (3.2914), `absences` (0.8977), `G1` (0.2955), `reason_home` (0.1555), `famrel` (0.1495).
- **Subgroup Fairness Audit (Student Sex)**:
  - Evaluated on the held-out test cohort ($N=79$):
    - *Female Students*: $N = 38$ (48.10%), $\text{MAE} = 1.342$, $\text{RMSE} = 2.355$, $R^2 = 0.730$.
    - *Male Students*: $N = 41$ (51.90%), $\text{MAE} = 0.977$, $\text{RMSE} = 1.491$, $R^2 = 0.879$.
  - Differences reflect sample-level variance on a small held-out test cohort rather than statistically verified bias. Outputs remain strictly advisory.
- **Key Insight**: Interim examination marks drastically elevate predictive accuracy, while pre-exam attributes serve as effective early screening indicators.

---

### 16. Screenshots Placeholder
```
[ Figure 1: Streamlit Home Dashboard with Global KPIs ]
[ Figure 2: Exploratory Data Analysis & Grade Distribution ]
[ Figure 3: Single Student Prediction & SHAP Waterfall Attribution ]
[ Figure 4: At-Risk Analysis & Automated Support Recommendations ]
[ Figure 5: Batch CSV Evaluation & Results Export ]
```

---

### 17. Testing
The project includes a comprehensive automated test suite with 16 test cases executed via Pytest:
- `test_data.py`: Verified dataset loading, schema dimensions, nulls, and duplicate absence.
- `test_preprocessing.py`: Verified feature engineering correctness and absence of G1/G2 in Exp A.
- `test_models.py`: Verified training execution, metric ranges, and file persistence.
- `test_predictions.py`: Verified single inference, uncertainty calculation, and batch CSV schema checks.
- `test_app.py`: Verified clean syntax and module imports.
- **Status**: 16 passed in 3.95s (100% pass rate).

---

### 18. Limitations
- Dataset originates from Portuguese secondary schools; transfer to Indian university engineering contexts requires local calibration.
- Sample size of 395 records limits deep learning feasibility.
- Single-semester cross-sectional snapshot rather than multi-year longitudinal tracking.

---

### 19. Conclusion
The developed system demonstrates that pairing supervised machine learning with Explainable AI enables transparent, defensible student early intervention systems. By rigorously isolating early pre-exam factors from mid-term evaluations, the platform establishes realistic expectations for institutional decision-makers.

---

### 20. Future Scope
- Integration with institutional Learning Management Systems (LMS) clickstream data.
- Longitudinal recurrent neural networks for multi-semester trajectory tracking.
- Interactive counterfactual explanations to provide prescriptive student action plans.

---

### 21. References
1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, Porto, Portugal, pp. 5–12. EUROSIS-ETI. Available at: http://hdl.handle.net/1822/8024
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 30)*, Long Beach, CA, pp. 4765–4774. Available at: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
3. Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. *WIREs Data Mining and Knowledge Discovery*, 10(3), e1355. https://doi.org/10.1002/widm.1355
4. Conati, C., Porayska-Pomsta, K., & Mavrikis, M. (2018). AI in Education needs interpretable machine learning: Lessons from Open Learner Modelling. In *Proceedings of the Third Annual Workshop on Human Interpretability in Machine Learning (WHI 2018)*, Stockholm, Sweden, pp. 21–27. https://doi.org/10.48550/arXiv.1807.00154
5. Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82–115. https://doi.org/10.1016/j.inffus.2019.12.012
6. Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In: Larusson, J. A., White, B. (eds) *Learning Analytics: From Research to Practice*, pp. 61–75. Springer, New York, NY. https://doi.org/10.1007/978-1-4614-3305-7_4
7. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd ed. Munich, Germany: Leanpub. Available at: https://christophm.github.io/interpretable-ml-book/
8. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. Available at: https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
