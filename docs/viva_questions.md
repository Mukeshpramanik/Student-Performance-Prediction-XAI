# Minor Research Project Viva Voce Preparation Guide
## 32 Comprehensive Questions & Detailed Technical Answers

---

### Category 1: Python & Software Engineering

#### Q1: Why did you choose Python for this project rather than R or Java?
**Answer:** Python provides an unmatched unified ecosystem spanning data wrangling (Pandas, NumPy), machine learning (Scikit-Learn), Explainable AI (SHAP), and rapid interactive dashboard development (Streamlit). Python also enables object-oriented modular design where data loaders, preprocessing pipelines, inference scripts, and web UI remain strictly decoupled.

#### Q2: How did you structure your codebase to follow professional software engineering practices?
**Answer:** We separated concerns into distinct folders: `config/` for path and seed constants, `data/` for raw and processed datasets, `src/` for core business logic, `models/` for serialized artifacts, `tests/` for automated Pytest verification, and `app.py` for presentation. We avoided hardcoded paths using Python's `pathlib.Path`.

#### Q3: Why is virtual environment management critical in machine learning projects?
**Answer:** A virtual environment isolates project dependencies, preventing version conflicts with system-level packages or other repositories. It guarantees reproducibility across different development workstations (e.g., MacBook Air running Apple Silicon).

---

### Category 2: Dataset & Exploratory Data Analysis

#### Q4: Describe the dataset used in this research.
**Answer:** We utilized the benchmark UCI Student Performance dataset (Cortez & Silva, 2008), focusing on the secondary school Mathematics course (`student-mat.csv`). It contains 395 student observations across 33 variables, including demographic attributes, parental education, behavioral habits (study time, alcohol consumption, absences), and periodic examination grades ($G1, G2, G3$).

#### Q5: Did you observe any missing values or duplicates in the dataset?
**Answer:** Automated inspection confirmed 0 missing values and 0 duplicate rows. Data types were validated, and final evaluation grades ($G3$) were verified to strictly fall within the authentic Portuguese 0–20 grading scale.

#### Q6: Why does final grade $G3$ exhibit a cluster of zeros?
**Answer:** During exploratory data analysis, 38 students (9.6%) scored 0 on the final evaluation ($G3$). The dataset records these as final evaluation marks of 0 without providing explicit administrative annotations for why zero marks were assigned.

#### Q7: What feature engineering did you introduce?
**Answer:** We engineered four domain-specific composite features:
1. `parent_edu_avg`: Mean of mother's and father's education levels.
2. `total_alc`: Weighted combined alcohol consumption index across workdays and weekends.
3. `study_travel_ratio`: Ratio of independent study time to travel time.
4. `academic_support_any`: Binary indicator denoting whether the student receives school or family educational support.
5. `at_risk`: Binary target flag defined as $G3 < 10$.

---

### Category 3: Machine Learning & Preprocessing

#### Q8: What is data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) is erroneously used to fit the model. We prevented leakage by performing train/test separation (80/20) *before* fitting any transformer. The `ColumnTransformer` (StandardScaler and OneHotEncoder) was strictly fitted on the training set and then applied to transform the test set.

#### Q9: What is the core difference between Experiment A and Experiment B?
**Answer:** 
- **Experiment A (Early Prediction)** strictly omits $G1$ and $G2$, relying purely on pre-examination background and behavioral factors. This simulates forecasting performance before any midterm exam takes place.
- **Experiment B (Mid-Term Prediction)** incorporates $G1$ and $G2$, reflecting the scenario when interim assessments have been evaluated.

#### Q10: Why did you use stratified splitting for classification?
**Answer:** The at-risk class ($G3 < 10$) accounts for 32.9% of the cohort. Stratified splitting ensures that both the training and testing subsets maintain this exact 32.9% class ratio, avoiding distribution shift.

#### Q11: Which regression models did you evaluate and why?
**Answer:** We benchmarked:
1. *Linear Regression*: Parametric baseline.
2. *Decision Tree Regressor*: Non-linear single-tree baseline.
3. *Random Forest Regressor*: Bagging ensemble to reduce variance.
4. *Gradient Boosting Regressor*: Sequential boosting ensemble to minimize residual error.

#### Q12: Which classification models did you evaluate and why?
**Answer:** We evaluated *Logistic Regression* (regularized linear classifier with high interpretability and direct risk probability estimation) and *Random Forest Classifier* (non-linear ensemble capturing complex feature interactions).

---

### Category 4: Evaluation Metrics & Findings

#### Q13: What metrics did you use to evaluate regression models?
**Answer:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$). MAE represents average error magnitude in grade points, while RMSE penalizes large outlier errors more heavily.

#### Q14: What were your actual regression results?
**Answer:** 
- In **Experiment A**: Random Forest was the best performer ($\text{MAE} = 3.15, \text{RMSE} = 3.94, R^2 = 0.241$).
- In **Experiment B**: Gradient Boosting achieved the highest performance ($\text{MAE} = 1.15, \text{RMSE} = 1.95, R^2 = 0.814$).

#### Q15: Why is $R^2$ in Experiment A significantly lower than Experiment B?
**Answer:** Pre-examination demographic and behavioral factors inherently carry high variance and stochasticity; they set baseline academic predispositions but cannot account for late-semester effort or exam mastery. Once $G1$ and $G2$ become available, they directly quantify subject comprehension, surging explained variance from 24.1% to 81.4%.

#### Q16: What were your actual classification results for at-risk detection?
**Answer:**
- In **Experiment A**: Logistic Regression achieved 69.62% accuracy and 0.711 ROC-AUC.
- In **Experiment B**: Logistic Regression achieved 91.14% accuracy, 95.24% precision, 76.92% recall, and 0.977 ROC-AUC.

#### Q17: What is 5-Fold Cross Validation and why was it employed?
**Answer:** 5-Fold CV partitions the training cohort into 5 equal folds, training on 4 and validating on the 5th across 5 iterations. This ensures performance estimates are robust against sampling variance and confirms that models are not overfitting to a single split.

---

### Category 5: Explainable AI & SHAP

#### Q18: What is SHAP and what mathematical concept is it based on?
**Answer:** SHAP (SHapley Additive exPlanations) is a game-theoretic approach to model interpretability developed by Lundberg & Lee (2017). It is based on Shapley values from cooperative game theory, where each feature value acts as a 'player' contributing to the final prediction 'payout'.

#### Q19: What properties make SHAP superior to standard feature importance methods like MDI?
**Answer:** SHAP uniquely satisfies three essential mathematical axioms:
1. *Local Accuracy*: The sum of feature attributions equals the difference between model output and base expectation.
2. *Missingness*: Features with zero impact receive zero attribution.
3. *Consistency*: If a model changes such that a feature's marginal contribution increases, its Shapley value cannot decrease. Mean Decrease in Impurity (MDI) lacks consistency and overvalues high-cardinality features.

#### Q20: Explain the difference between global and local SHAP explanations.
**Answer:**
- *Global Explanations* (beeswarm and bar plots) summarize overall feature importance across the entire dataset cohort.
- *Local Explanations* (waterfall plots) unpack a single student's prediction, showing exactly how each specific attribute nudged their score above or below the cohort baseline.

#### Q21: Which features had the highest global importance in Experiment A?
**Answer:** Quantified using mean absolute SHAP values, the top features in Experiment A (Random Forest) are: `absences` (0.9813), `failures` (0.9097), `health` (0.1606), `goout` (0.1485), and `parent_edu_avg` (0.1403). These reflect average magnitude of observational attribution across the cohort, not causal percentages.

#### Q22: Which features dominated in Experiment B?
**Answer:** In Experiment B (Gradient Boosting), interim assessment features dominated in mean absolute SHAP importance: second period grade `G2` (3.2914), followed by `absences` (0.8977), first period grade `G1` (0.2955), `reason_home` (0.1555), and `famrel` (0.1495).

---

### Category 6: Fairness, Ethics & System Implementation

#### Q23: How did you evaluate algorithmic fairness in this project?
**Answer:** We conducted an empirical subgroup audit across student biological sex on the held-out test cohort ($N=79$):
- Female students ($N=38$, 48.10%): $\text{MAE} = 1.342$, $\text{RMSE} = 2.355$, $R^2 = 0.730$.
- Male students ($N=41$, 51.90%): $\text{MAE} = 0.977$, $\text{RMSE} = 1.491$, $R^2 = 0.879$.
We interpret these observed sample variations cautiously due to small subgroup sample sizes; they do not constitute statistically verified bias. Crucially, demographic attributes must never be used in isolation for automated academic penalties.

#### Q24: What ethical safeguards did you implement?
**Answer:** 
1. The system is explicitly defined as a *decision-support tool*, never an automated decision-maker.
2. Sensitive demographic features are prohibited from serving as standalone criteria for punitive academic actions.
3. Tree-ensemble spread estimates are presented to communicate model variance transparently.

#### Q25: How does the rule-based recommendation engine work?
**Answer:** The recommendation engine parses validated student inputs and triggers targeted, supportive guidance. For example, absences $\ge 10$ prompt attendance reviews; study time $< 2$ hours/week prompts structured study planning; prior failures prompt foundational diagnostic tutoring. All suggestions are framed non-judgmentally.

#### Q26: How does batch CSV prediction work in your Streamlit dashboard?
**Answer:** Users upload a CSV file. The backend validates column presence, extracts engineered features, executes vectorized preprocessing via the saved `ColumnTransformer`, generates batch predictions and risk categories, and allows users to download the enriched CSV.

#### Q27: How is prediction uncertainty quantified for a single student?
**Answer:** For tree ensemble models, we compute the empirical standard deviation across individual decision trees in the ensemble ($\sigma_{\text{trees}}$) and derive an empirical spread estimate $[\hat{y} - 1.96\sigma, \hat{y} + 1.96\sigma]$, bounded between 0 and 20. This communicates model variance across trees rather than a statistically calibrated confidence interval.

---

### Category 7: Research Validity & Limitations

#### Q28: Can your model guarantee academic success for a student?
**Answer:** No. Machine learning models generate probabilistic estimates based on historical correlation. They cannot account for unobserved personal events, sudden health crises, or qualitative factors.

#### Q29: Can this model be directly deployed in an Indian university engineering department?
**Answer:** No. The dataset represents Portuguese secondary school mathematics students. Indian engineering curricula have different grading scales, attendance mandates, and demographic dynamics. The methodology transfers directly, but the models must be retrained on local institutional data.

#### Q30: How did you ensure software quality in your implementation?
**Answer:** We authored an automated Pytest test suite containing 16 unit tests across 5 test modules (`test_data.py`, `test_preprocessing.py`, `test_models.py`, `test_predictions.py`, `test_app.py`), achieving a 100% pass rate.

#### Q31: What would be your next step if you had 6 more months on this project?
**Answer:** I would integrate real-time clickstream telemetry from Learning Management Systems (Moodle/Canvas), incorporate longitudinal trajectory models (LSTMs/Transformers), and build counterfactual 'What-if' prescriptive explanations.

#### Q32: What is the single most important academic contribution of your project?
**Answer:** Demonstrating that Explainable AI transforms machine learning from an untrusted academic grading black box into an open, interpretable pedagogical partner that empowers educators to intervene early with compassionate, targeted support.
