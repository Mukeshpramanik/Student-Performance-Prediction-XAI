# An Explainable Machine Learning Framework for Early Prediction and Analysis of Student Academic Performance Using Learning Analytics

**Author:** Mukesh Pramanik  
**Academic Degree:** Bachelor of Technology in Computer Science & Engineering  
**Department:** Computer Science & Engineering  
**Project Repository:** https://github.com/Mukeshpramanik/Student-Performance-Prediction-XAI  

---

## 1. Title
**An Explainable Machine Learning Framework for Early Prediction and Analysis of Student Academic Performance Using Learning Analytics**

---

## 2. Abstract
Timely identification of students requiring academic support is essential for mitigating course attrition and fostering inclusive educational outcomes. While supervised machine learning algorithms demonstrate high diagnostic accuracy on educational datasets, conventional predictive architectures operate as opaque 'black boxes', limiting practical adoption by educators and academic advisors. This research develops a reproducible, end-to-end Explainable Artificial Intelligence (XAI) framework for student performance prediction and early intervention. Utilizing the benchmark UCI Student Performance dataset (395 secondary school students in mathematics), we structure two distinct empirical evaluations: **Experiment A (Early Prediction)**, leveraging solely pre-examination demographic, familial, and behavioral attributes; and **Experiment B (Mid-Term Prediction)**, integrating interim examination evaluations ($G1$ and $G2$). 

Our findings indicate that pre-examination features provide a vital early screening signal, with Random Forest achieving a test Mean Absolute Error (MAE) of 3.15 and $R^2$ of 0.241, while Logistic Regression obtains an At-Risk ($G3 < 10$) detection ROC-AUC of 0.711. Incorporating mid-term academic evaluations causes predictive power to surge, with Gradient Boosting Regressor achieving an MAE of 1.15, RMSE of 1.95, and $R^2$ of 0.814, and Logistic Regression attaining an accuracy of 91.14% and ROC-AUC of 0.977. Using SHAP (SHapley Additive exPlanations), we demystify model outputs at both global cohort levels and individual student levels through localized waterfall plots. We further conduct subgroup audits across student biological sex to evaluate error distributions. The framework is deployed as a responsive, open-source Streamlit dashboard integrating automated, non-judgmental remediation advice, bridging the gap between academic analytics and actionable pedagogical counseling.

---

## 3. Keywords
Learning Analytics; Educational Data Mining; Explainable Artificial Intelligence (XAI); SHAP; Student Performance Prediction; Academic Early Warning Systems; Machine Learning.

---

## 4. Introduction
Higher educational institutions and secondary schools globally grapple with student retention, course failure rates, and academic disengagement. Early identification of academic struggle is a prerequisite for administering targeted pedagogical interventions. Historically, academic assessment has been reactive: instructors detect underperformance primarily after summative examinations conclude, leaving limited temporal margin for remediation.

The advent of Learning Analytics (LA) and Educational Data Mining (EDM) offers a proactive paradigm. By analyzing background demographic indicators, study routines, past academic trajectory, and interim grades, machine learning algorithms can forecast final grades and classify students into risk tiers. However, deployment in institutional governance faces two major challenges:
1. **The Temporal Availability Dilemma**: Predictive performance differs drastically depending on whether models rely exclusively on early baseline characteristics or incorporate mid-semester examination grades. Conflating these two stages produces misleading pedagogical expectations.
2. **The Black-Box Dilemma**: Complex ensemble models (e.g., Random Forests, Gradient Boosted Trees) provide superior predictive accuracy over simple linear models but fail to articulate *why* a particular student is designated as academically vulnerable. Educational advisors cannot ethically mandate remedial interventions without clear, interpretable rationale.

This study directly resolves both dilemmas by presenting a rigorous comparative framework separating early and mid-term prediction, enriched with SHAP-based game-theoretic explainability.

---

## 5. Background
Educational Data Mining sits at the intersection of computer science, statistics, and education. Early EDM models relied on classical statistical regression to identify macro-level risk indicators, such as socioeconomic status and parental education. With modern machine learning, researchers have adopted non-parametric decision trees, support vector machines, and boosting algorithms. 

Despite performance gains, stakeholders frequently reject algorithmic decisions when models fail to convey interpretability. Explainable AI (XAI) addresses this barrier by generating post-hoc, model-agnostic explanations. In particular, Lundberg and Lee's SHAP framework translates Shapley values from cooperative game theory into additive feature attributions, providing both global factor ranking and exact sample-level attribution.

---

## 6. Literature Review
Academic literature on student performance modeling reflects an evolution from descriptive analytics to interpretable predictive architectures:

1. **Cortez and Silva (2008)** conducted a pioneering study utilizing the UCI Student Performance dataset across Mathematics and Portuguese subjects. Benchmarking Linear Regression, Decision Trees, Neural Networks, and Support Vector Machines, they observed that secondary school grades are strongly influenced by past failure history and previous evaluations ($G1$ and $G2$).
2. **Romero and Ventura (2020)** surveyed two decades of Educational Data Mining and Learning Analytics, underscoring that the primary bottleneck in contemporary LA is not algorithmic accuracy, but pedagogical interpretability and teacher adoption.
3. **Lundberg and Lee (2017)** formulated the unified SHAP framework, demonstrating that Shapley values represent the unique class of additive feature attributions satisfying local accuracy, missingness, and consistency properties.
4. **Conati, Porayska-Pomsta, and Mavrikis (2018)** emphasized that pedagogical trust requires transparent models that explicitly justify their suggestions to teaching staff.
5. **Barredo Arrieta et al. (2020)** surveyed XAI methodologies, classifying explainability into transparent intrinsic models and post-hoc model-agnostic interpretability tools.

---

## 7. Research Gap
Despite extensive EDM research, three notable gaps persist in the literature:
1. **Temporal Conflation**: Studies regularly report stellar predictive accuracies ($R^2 > 0.85$) without explicitly separating features available at course commencement from intermediate exam marks that only appear midway through the semester.
2. **Absence of Localized Student-Level Explanations**: Existing implementations predominantly output global feature importance charts (e.g., standard MDI tree importances), failing to explain why an *individual* student was flagged as vulnerable.
3. **Disconnection from Actionable Support**: Published models often operate as theoretical exercises without deployable interfaces or concrete, compassionate pedagogical recommendations.

---

## 8. Research Objectives
This study addresses these gaps through the following objectives:
- **RO1**: Develop and evaluate multiple regression algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) for predicting final numerical grades ($G3$).
- **RO2**: Develop and evaluate binary classification models (Logistic Regression, Random Forest Classifier) for detecting students at risk of failure ($G3 < 10$).
- **RO3**: Isolate and quantify the predictive impact of intermediate evaluation marks via dual experiments (Experiment A: Early vs. Experiment B: Mid-Term).
- **RO4**: Implement global and local SHAP explainability to provide transparent feature attribution for educators.
- **RO5**: Conduct a subgroup performance audit across biological sex to investigate demographic equity.
- **RO6**: Build a deployable, open-source Streamlit application with rule-based intervention advice.

---

## 9. Research Questions
- **RQ1**: Which machine learning algorithm provides the most suitable predictive accuracy for student academic grade prediction in early vs. mid-term scenarios?
- **RQ2**: What quantitative difference in explanatory variance ($R^2$, MAE, ROC-AUC) results from incorporating first and second period exam marks ($G1, G2$)?
- **RQ3**: Which demographic, academic, and behavioral features demonstrate the strongest global influence on model predictions?
- **RQ4**: How do individualized SHAP waterfall attributions enhance pedagogical utility compared to standard global rankings?
- **RQ5**: How do model performance and error distributions vary across student demographic subgroups (e.g., biological sex)?

---

## 10. Dataset Description
We utilize the official UCI Student Performance Dataset (Mathematics course: `student-mat.csv`), collected by Cortez and Silva (2008) from two Portuguese public secondary schools: Gabriel Pereira (GP) and Mousinho da Silveira (MS). 

The dataset contains $N = 395$ student instances and 33 original attributes:
- **Demographic Attributes**: `school` (GP/MS), `sex` (F/M), `age` (15-22), `address` (Urban/Rural), `famsize` (<=3 / >3), `Pstatus` (Together/Apart).
- **Family Background**: `Medu` and `Fedu` (0-4), `Mjob` and `Fjob` (nominal), `guardian` (mother/father/other).
- **Academic Environment**: `reason` (school choice), `traveltime` (1-4), `studytime` (1-4), `failures` (0-3 past class failures), `schoolsup` (binary support), `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`.
- **Social & Behavioral**: `romantic`, `famrel` (1-5), `freetime` (1-5), `goout` (1-5), `Dalc` (workday alcohol 1-5), `Walc` (weekend alcohol 1-5), `health` (1-5), `absences` (0-93).
- **Academic Evaluations**: `G1` (first period score: 0-20), `G2` (second period score: 0-20), `G3` (final evaluation score: 0-20, primary target).

Dataset integrity checks confirmed **0 missing values** and **0 duplicate records**.

---

## 11. Proposed Methodology
The end-to-end framework comprises six sequential stages:
1. **Data Ingestion & Integrity Validation**: Automated schema inspection and bounding checks.
2. **Domain Feature Engineering**: Formulation of composite educational indicators:
   - Parental Education Average: $\text{parent\_edu\_avg} = \frac{\text{Medu} + \text{Fedu}}{2}$
   - Total Alcohol Index: $\text{total\_alc} = \frac{5 \cdot \text{Dalc} + 2 \cdot \text{Walc}}{7}$
   - Study-to-Travel Ratio: $\text{study\_travel\_ratio} = \frac{\text{studytime}}{\text{traveltime} + 0.1}$
   - Academic Support Indicator: $\text{academic\_support\_any} = \mathbf{1}(\text{schoolsup} = \text{'yes'} \lor \text{famsup} = \text{'yes'})$
   - At-Risk Binary Indicator: $\text{at\_risk} = \mathbf{1}(G3 < 10)$
3. **Leakage-Free Preprocessing**: Splitting train (80%) and test (20%) partitions prior to fitting `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals).
4. **Dual Experimental Training**: Separate pipeline branches for Experiment A (Early) and Experiment B (Mid-Term).
5. **Explainability Extraction**: SHAP TreeExplainer and LinearExplainer applied to test samples.
6. **Application Deployment**: Interactive Streamlit user interface.

---

## 12. Data Preprocessing
To strictly prevent data leakage:
- Preprocessing pipelines were fitted solely on the training partition ($N_{\text{train}} = 316$) and used to transform the testing partition ($N_{\text{test}} = 79$).
- One-hot encoding was initialized with `handle_unknown="ignore"` to safeguard against runtime errors during prospective inference.
- Classification train/test splits utilized stratified sampling based on the `at_risk` target, guaranteeing identical class proportions (32.9% at-risk) across splits.

---

## 13. Exploratory Data Analysis
Key statistical and visual patterns observed during EDA across the 395 students in mathematics:
- **Grade Distribution (Figure 1)**: Final grade ($G3$) has a mean of 10.42 ($\pm 4.58$, range 0–20). A notable subset of 38 students (9.6%) scored 0. Period grades $G1$ (mean: 10.91) and $G2$ (mean: 10.71) display approximately normal distributions with minor left-skew.
- **Correlation Structure (Figure 2)**: Strong positive correlations exist between consecutive evaluations ($r_{G1,G2} = 0.85$, $r_{G2,G3} = 0.90$). Demographic and behavioral indicators show moderate pairwise correlation with $G3$, with `failures` ($r = -0.36$), `Medu` ($r = 0.22$), and `studytime` ($r = 0.10$) exhibiting direct associations.
- **Study Time Pattern (Figure 3)**: Modest positive correlation with final grade ($r = 0.098, p = 0.052$); students studying $<2$ hours/week averaged $10.05$ compared to $11.40$ for those studying $5–10$ hours/week.
- **Absences Impact (Figure 4)**: Displayed a non-linear, right-skewed distribution (median: 4, max: 75; $r = 0.034$); students with $>10$ absences exhibited an at-risk rate of 45.5% compared to 30.4% for those with $\le 10$ absences.
- **Interim Examination Association (Figure 5)**: Bivariate scatter plots demonstrate that student grade trajectories stabilize significantly by the second period ($G2$), with minimal grade recovery observed among students failing both $G1$ and $G2$.
- **Past Failures and Risk Incidence (Figure 6)**: Prior class failures serves as the single strongest pre-exam vulnerability indicator ($r = -0.36$); students with 1 or more past failures exhibited a 62.7% at-risk incidence ($G3 < 10$), compared to 25.0% for students with zero failures (a $\approx 2.5\times$ increase).

*Note: All corresponding publication-quality figures are cataloged in `reports/figures/` (Figures 1–6).*

---

## 14. Machine Learning Models
We benchmarked four regression and two classification architectures:
- **Linear Regression**: Classical ordinary least squares baseline.
- **Decision Tree Regressor**: Non-linear baseline with `max_depth=5` to prevent overfitting.
- **Random Forest Regressor / Classifier**: Ensemble bagging architecture (100 estimators, `max_depth=6`).
- **Gradient Boosting Regressor**: Sequential boosting architecture (100 estimators, learning rate $\eta = 0.05$).
- **Logistic Regression**: Regularized generalized linear classifier optimizing cross-entropy loss.

---

## 15. Model Evaluation Metrics
Performance was evaluated across standardized metrics:
- **Continuous Metrics**:
  - Mean Absolute Error: $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$
  - Root Mean Squared Error: $\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
  - Coefficient of Determination: $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Classification Metrics**:
  - Accuracy, Precision, Recall, F1-Score, and Receiver Operating Characteristic Area Under Curve (ROC-AUC).
- **Validation Protocol**: 5-Fold Cross Validation on the training cohort alongside held-out test evaluation.

---

## 16. Experimental Results

### 16.1 Regression Benchmarks
The empirical results across held-out test sets ($N_{\text{test}} = 79$) and 5-fold cross-validation on the training set ($N_{\text{train}} = 316$) are summarized in Table 1:

#### Table 1: Regression Model Evaluation Benchmarks across Held-out Test Cohort and 5-Fold Cross-Validation

| Experiment | Model | MAE | RMSE | $R^2$ Score | 5-Fold CV MAE | 5-Fold CV $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A (Early)** | Linear Regression | 3.42 | 4.24 | 0.121 | 3.54 $\pm$ 0.39 | -0.113 $\pm$ 0.254 |
| **Exp A (Early)** | Decision Tree | 3.45 | 4.52 | 0.002 | 3.52 $\pm$ 0.38 | -0.155 $\pm$ 0.150 |
| **Exp A (Early)** | **Random Forest** | **3.15** | **3.94** | **0.241** | **3.00 $\pm$ 0.36** | **0.192 $\pm$ 0.126** |
| **Exp A (Early)** | Gradient Boosting | 3.21 | 4.01 | 0.215 | 3.14 $\pm$ 0.37 | 0.157 $\pm$ 0.093 |
| **Exp B (Mid-Term)** | Linear Regression | 1.66 | 2.41 | 0.718 | 1.39 $\pm$ 0.24 | 0.805 $\pm$ 0.031 |
| **Exp B (Mid-Term)** | Decision Tree | 1.47 | 2.65 | 0.657 | 1.05 $\pm$ 0.17 | 0.829 $\pm$ 0.052 |
| **Exp B (Mid-Term)** | Random Forest | 1.20 | 1.98 | 0.808 | 0.97 $\pm$ 0.22 | 0.883 $\pm$ 0.043 |
| **Exp B (Mid-Term)** | **Gradient Boosting** | **1.15** | **1.95** | **0.814** | **1.01 $\pm$ 0.21** | **0.878 $\pm$ 0.042** |

### 16.2 Classification Benchmarks (At-Risk: $G3 < 10$)
The empirical classification metrics for identifying at-risk students are summarized in Table 2:

#### Table 2: Classification Model Evaluation Benchmarks for Academic At-Risk Screening

| Experiment | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A (Early)** | **Logistic Regression** | **69.62%** | **57.14%** | **30.77%** | **0.400** | **0.711** |
| **Exp A (Early)** | Random Forest Classifier | 69.62% | 62.50% | 19.23% | 0.294 | 0.684 |
| **Exp B (Mid-Term)** | **Logistic Regression** | **91.14%** | **95.24%** | 76.92% | **0.851** | **0.977** |
| **Exp B (Mid-Term)** | Random Forest Classifier | 89.87% | 84.62% | **84.62%** | 0.846 | 0.961 |

---

## 17. Explainable AI Using SHAP
Model interpretability was evaluated using SHAP (SHapley Additive exPlanations) to measure the relative contribution of input features to model predictions. Feature importance is quantified using mean absolute SHAP values ($\frac{1}{n} \sum_{i=1}^n |\phi_i^{(j)}|$), reflecting the average magnitude of a feature's attribution on the model's output across the test cohort. These values represent observational attribution within the trained model and should not be interpreted as percentage shares of total variance or as causal effects.

### 17.1 Global Feature Importance
- **Experiment A (Early Prediction — Random Forest Regressor, Figures 7 & 8)**:
  1. `absences`: Mean absolute SHAP value = 0.9813
  2. `failures`: Mean absolute SHAP value = 0.9097
  3. `health`: Mean absolute SHAP value = 0.1606
  4. `goout`: Mean absolute SHAP value = 0.1485
  5. `parent_edu_avg`: Mean absolute SHAP value = 0.1403

In the absence of examination marks, student absenteeism and past academic failures emerge as the primary predictive drivers, followed by self-reported health status, social activity frequency (`goout`), and average parental education level (`parent_edu_avg`). Corresponding global visualizations are provided in Figure 7 (beeswarm summary) and Figure 8 (mean absolute attribution bar chart).

- **Experiment B (Mid-Term Prediction — Gradient Boosting Regressor, Figures 9 & 10)**:
  1. `G2`: Mean absolute SHAP value = 3.2914
  2. `absences`: Mean absolute SHAP value = 0.8977
  3. `G1`: Mean absolute SHAP value = 0.2955
  4. `reason_home`: Mean absolute SHAP value = 0.1555
  5. `famrel`: Mean absolute SHAP value = 0.1495

When interim grades are provided, the second period examination grade ($G2$) exhibits the highest mean absolute attribution (3.2914), followed by school absences (0.8977), first period grade ($G1$, 0.2955), proximity-based school choice (`reason_home`, 0.1555), and family relationship quality (`famrel`, 0.1495). Corresponding global visualizations are presented in Figure 9 (beeswarm summary) and Figure 10 (mean absolute attribution bar chart).

*Note: All corresponding publication plots are cataloged in `reports/figures/` (Figure 7: `07_shap_summary_exp_a.png`, Figure 8: `08_shap_bar_exp_a.png`, Figure 9: `09_shap_summary_exp_b.png`, Figure 10: `10_shap_bar_exp_b.png`).*

### 17.2 Local Prediction Explanations
At the individual student level, SHAP explains individual predictions by decomposing the difference between the model forecast $\hat{y}$ and the baseline expected value $E[f(X)]$ across the training distribution into additive feature attributions:
$$\hat{y} = E[f(X)] + \sum_{j=1}^M \phi_j$$
Local waterfall visualizations depict the directional impact of individual feature values—identifying which specific attributes positively elevate or negatively depress a student's predicted grade—providing educators with case-specific context for targeted academic mentoring.

---

## 18. Fairness and Subgroup Analysis
To examine model performance consistency across demographic segments, an empirical subgroup evaluation was conducted across student biological sex on the held-out test cohort ($N_{\text{test}} = 79$) using the primary mid-term model (Gradient Boosting Regressor), as reported in Table 3:

#### Table 3: Subgroup Performance Evaluation Across Student Biological Sex (Gradient Boosting Regressor, Held-out Test Cohort $N_{\text{test}} = 79$)

| Subgroup | Sample Size ($N$) | Proportion | Subgroup MAE | Subgroup RMSE | Subgroup $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Female Students** | 38 | 48.10% | 1.342 | 2.355 | 0.730 |
| **Male Students** | 41 | 51.90% | 0.977 | 1.491 | 0.879 |

### 18.1 Subgroup Interpretation and Ethical Considerations
- On the held-out test sample, lower prediction errors were observed for male students ($\text{MAE} = 0.977$, $R^2 = 0.879$) compared to female students ($\text{MAE} = 1.342$, $R^2 = 0.730$).
- These numerical differences must be interpreted with caution. The test cohort sample sizes ($N=38$ female, $N=41$ male) are small, meaning observed differences reflect sample-level variance rather than statistically verified bias or systemic algorithmic disparity. Formal statistical significance testing was not established on this sample size.
- Consequently, demographic attributes such as sex, home address, or parental background must **never** be used in isolation to make automated academic judgments, restrict educational opportunities, or assign disciplinary actions. Predictions are strictly advisory decision-support estimates.

---

## 19. Discussion
Addressing the research questions formulated for this study:
- **RQ1 & RQ2**: In early prediction without prior exam marks, Random Forest achieved the most favorable regression performance ($R^2 = 0.241$, $\text{MAE} = 3.15$), and Logistic Regression provided the most balanced screening classification ($\text{ROC-AUC} = 0.711$). Incorporating mid-term grades ($G1, G2$) substantially increased predictive performance, with Gradient Boosting reaching $R^2 = 0.814$ ($\text{MAE} = 1.15$) and Logistic Regression reaching $\text{ROC-AUC} = 0.977$. This confirms that pre-exam demographic and behavioral factors provide an initial screening baseline, whereas recent academic evaluations are necessary for precise summative forecasting.
- **RQ3 & RQ4**: Mean absolute SHAP analysis demonstrated that pre-examination predictions are primarily driven by `absences` (0.9813) and `failures` (0.9097), alongside `health`, `goout`, and `parent_edu_avg`. In mid-term prediction, `G2` (3.2914) and `absences` (0.8977) exhibit the highest attribution, followed by `G1` (0.2955), `reason_home` (0.1555), and `famrel` (0.1495). Local SHAP waterfall plots provide additive individual-level breakdowns without relying on unverified aggregate percentage claims.
- **RQ5**: Subgroup evaluation across student sex identified test-sample variations ($\text{MAE} = 1.342$ for females vs. $0.977$ for males). Given the modest sample size, these differences cannot be construed as confirmed bias, underscoring the requirement that model outputs remain strictly advisory within educational workflows.

---

## 20. Limitations
1. **Dataset Geography**: The data stems from two secondary schools in Alentejo, Portugal. Transferability to university-level engineering students in other geographical contexts requires local retraining.
2. **Cohort Size**: With 395 student instances, deep learning architectures are unfeasible without risking severe overfitting. Classical ML models are best suited for this scale.
3. **Cross-Sectional Scope**: The data captures a single academic session; longitudinal tracking across multiple consecutive semesters would provide richer temporal dynamics.

---

## 21. Conclusion
This minor research project successfully designed, implemented, and verified an explainable machine learning framework for student performance prediction. By distinguishing early pre-exam prediction from mid-term evaluation, we provide a realistic, leakage-free benchmark for educational analytics. The integration of game-theoretic SHAP interpretability and a user-friendly Streamlit dashboard transforms raw predictive outputs into defensible, actionable decision-support tools for academic mentors.

---

## 22. Future Scope
- **LMS Clickstream Ingestion**: Integrating real-time logs from Moodle or Canvas (lecture video completion, forum participation, assignment turnaround).
- **Sequential Neural Modeling**: Applying Recurrent Neural Networks (LSTMs) or Transformers across multi-semester university curricula.
- **Counterfactual XAI**: Developing prescriptive 'What-if' engines to illustrate the minimal behavioral adjustments required to transition a student from 'At-Risk' to 'Good Standing'.

---

## 23. References
1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, Porto, Portugal, pp. 5–12. EUROSIS-ETI. Available at: http://hdl.handle.net/1822/8024
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 30)*, Long Beach, CA, pp. 4765–4774. Available at: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
3. Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. *WIREs Data Mining and Knowledge Discovery*, 10(3), e1355. https://doi.org/10.1002/widm.1355
4. Conati, C., Porayska-Pomsta, K., & Mavrikis, M. (2018). AI in Education needs interpretable machine learning: Lessons from Open Learner Modelling. In *Proceedings of the Third Annual Workshop on Human Interpretability in Machine Learning (WHI 2018)*, Stockholm, Sweden, pp. 21–27. https://doi.org/10.48550/arXiv.1807.00154
5. Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., ... & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82–115. https://doi.org/10.1016/j.inffus.2019.12.012
6. Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In: Larusson, J. A., White, B. (eds) *Learning Analytics: From Research to Practice*, pp. 61–75. Springer, New York, NY. https://doi.org/10.1007/978-1-4614-3305-7_4
7. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd ed. Munich, Germany: Leanpub. Available at: https://christophm.github.io/interpretable-ml-book/
8. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. Available at: https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
