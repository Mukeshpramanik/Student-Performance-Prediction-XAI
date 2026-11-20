# Methodology: Dual-Stage Explainable Machine Learning Framework

## Overview
This document details the rigorous experimental design and statistical preprocessing pipeline applied in the *Explainable AI-Based Student Academic Performance Prediction and Early Intervention System*.

---

## 1. Experimental Design: Early vs. Mid-Term Prediction
To resolve the temporal availability dilemma common in Educational Data Mining, we structure two distinct, non-overlapping experimental configurations:

### Experiment A: Early Prediction Pipeline
- **Temporal Context**: Commencement of the academic term prior to any formal examination evaluations.
- **Available Features**: 17 categorical attributes + 17 numeric attributes (including demographic information, family background, social habits, prior failures, and school absences).
- **Excluded Features**: $G1$ (first period grade) and $G2$ (second period grade).
- **Pedagogical Objective**: Broad-scale risk screening to prioritize high-risk candidates for early institutional counseling before midterm exams occur.

### Experiment B: Mid-Term Prediction Pipeline
- **Temporal Context**: Middle of the academic term after first and second period examinations have been graded.
- **Available Features**: All Experiment A features plus $G1$ and $G2$.
- **Pedagogical Objective**: High-precision academic forecasting and intervention refinement prior to final summative assessments ($G3$).

---

## 2. Leakage-Free Preprocessing Pipeline
To guarantee complete experimental validity and prevent test data contamination:
1. **Partitioning**: The cleaned dataset ($N=395$) is split into an 80% training set ($N=316$) and a 20% test set ($N=79$) with a fixed random seed (`seed=42`).
2. **ColumnTransformer Fitting**: 
   - `StandardScaler` is applied to all numeric columns.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` is applied to all categorical columns.
   - Transformers are strictly fitted on the training split only and applied downstream to the testing split.
3. **Stratified Sampling for Classification**:
   - For binary at-risk classification ($G3 < 10$), `train_test_split` uses `stratify=y` to preserve the 32.9% class balance across both partitions.

---

## 3. Algorithm Selection & Optimization
- **Linear Regression**: Ordinary Least Squares baseline establishing linear relationships.
- **Decision Tree Regressor**: Non-linear baseline constrained with `max_depth=5` to prevent memorization of small training cohorts.
- **Random Forest (Regressor & Classifier)**: Ensemble bagging with 100 decision trees, capturing complex non-linear feature interactions while mitigating variance.
- **Gradient Boosting Regressor**: Gradient-boosted decision trees with 100 boosting stages and learning rate $\eta = 0.05$.
- **Logistic Regression**: L2-regularized logistic classification with maximum iterations set to 1000.

---

## 4. Cross-Validation Protocol
- **Regression**: 5-Fold Cross-Validation on the training cohort evaluating Mean Absolute Error (MAE) and $R^2$.
- **Classification**: Stratified 5-Fold Cross-Validation evaluating Accuracy, F1-Score, and ROC-AUC.
