"""
Explainable AI-Based Student Academic Performance Prediction and Early Intervention System
Main Interactive Streamlit Dashboard Application
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, MODELS_DIR,
    FIGURES_DIR, METRICS_DIR, TABLES_DIR, AT_RISK_THRESHOLD
)
from src.data_loader import load_raw_data, load_processed_data, validate_dataset
from src.feature_engineering import add_engineered_features
from src.predictions import (
    predict_single_student, predict_batch_csv,
    generate_sample_template_csv, load_artifacts
)
from src.explainability import plot_local_waterfall, plot_summary, plot_global_bar

# Configure Streamlit page
st.set_page_config(
    page_title="Student Academic XAI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    """Load and cache processed student data."""
    if PROCESSED_DATA_FILE.exists():
        return load_processed_data(PROCESSED_DATA_FILE)
    elif RAW_DATA_FILE.exists():
        raw_df = load_raw_data(RAW_DATA_FILE)
        return add_engineered_features(raw_df)
    else:
        # Fallback empty dataframe
        return pd.DataFrame()


@st.cache_data
def get_metrics_tables():
    """Load saved evaluation metrics."""
    reg_path = METRICS_DIR / "regression_metrics.csv"
    clf_path = METRICS_DIR / "classification_metrics.csv"
    cv_reg_path = METRICS_DIR / "cv_regression_metrics.csv"
    cv_clf_path = METRICS_DIR / "cv_classification_metrics.csv"

    reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
    clf_df = pd.read_csv(clf_path) if clf_path.exists() else pd.DataFrame()
    cv_reg = pd.read_csv(cv_reg_path) if cv_reg_path.exists() else pd.DataFrame()
    cv_clf = pd.read_csv(cv_clf_path) if cv_clf_path.exists() else pd.DataFrame()

    return reg_df, clf_df, cv_reg, cv_clf


df = get_dataset()
reg_metrics, clf_metrics, cv_reg_metrics, cv_clf_metrics = get_metrics_tables()

# Sidebar Navigation
st.sidebar.markdown("## 🎓 Academic XAI System")
st.sidebar.markdown("**Decision-Support Platform**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home Dashboard",
        "Dataset Explorer",
        "Exploratory Data Analysis",
        "Student Performance Prediction",
        "At-Risk Analysis & Guidance",
        "Explainable AI (SHAP)",
        "Model Comparison & Benchmarks",
        "Batch CSV Prediction",
        "About Project & Ethics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Minor Research Project | B.Tech CSE")
st.sidebar.caption("Developer: Mukesh Pramanik")


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "Home Dashboard":
    st.markdown('<div class="main-title">Explainable AI-Based Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Research-Oriented Learning Analytics Decision-Support System</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dataset Records</div>
            <div class="metric-value">{len(df) if not df.empty else 395}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Attributes</div>
            <div class="metric-value">{df.shape[1] if not df.empty else 33}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Midterm R²</div>
            <div class="metric-value">0.814</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Midterm ROC-AUC</div>
            <div class="metric-value">0.977</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Executive Overview")
    st.info("""
    **Core Purpose**: This application provides early, interpretable predictions of student academic performance to facilitate proactive pedagogical interventions.
    It rigorously distinguishes between **Experiment A (Early Prediction)**—relying solely on pre-examination background factors—and **Experiment B (Mid-Term Prediction)**—integrating interim evaluation grades (G1 and G2).
    Every prediction is explained using **SHAP (SHapley Additive exPlanations)** to ensure fairness, transparency, and educator trust.
    """)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🎯 Primary Research Objectives")
        st.markdown("""
        - **Accurate Grade Estimation**: Continuous grade prediction (0–20) using tuned ensemble architectures.
        - **At-Risk Identification**: Binary early warning detection for students scoring below the pass mark ($G3 < 10$).
        - **Transparent Interpretability**: Global feature ranking and localized SHAP waterfall attribution for individual students.
        - **Empathetic Intervention Engine**: Automated, non-judgmental guidance addressing attendance, study habits, and remedial support.
        """)
    with col_right:
        st.markdown("#### 🔬 Key Analytical Findings")
        st.markdown("""
        - **Temporal Grade Availability**: Pre-examination indicators account for ~24% of grade variance ($R^2 = 0.24$), whereas adding midterm scores surges explained variance to >81% ($R^2 = 0.814$).
        - **Primary Early Predictors**: Past class failures, absenteeism, and weekly study time dominate early risk screening.
        - **Intervention Window**: Proactive support delivered before midterm exams provides the highest pedagogical impact.
        """)


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif page == "Dataset Explorer":
    st.markdown('<div class="main-title">UCI Student Performance Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive exploration and schema documentation for the Mathematics cohort</div>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset not found. Please verify `data/processed/student_clean.csv`.")
    else:
        st.markdown("### 🔍 Dataset Search & Filter")
        c1, c2, c3 = st.columns(3)
        with c1:
            school_filter = st.multiselect("Filter School", options=df["school"].unique(), default=df["school"].unique())
        with c2:
            sex_filter = st.multiselect("Filter Sex", options=df["sex"].unique(), default=df["sex"].unique())
        with c3:
            at_risk_filter = st.selectbox("At-Risk Status Filter", ["All Students", "At Risk (G3 < 10)", "Not At Risk (G3 >= 10)"])

        filtered_df = df[(df["school"].isin(school_filter)) & (df["sex"].isin(sex_filter))]
        if at_risk_filter == "At Risk (G3 < 10)":
            filtered_df = filtered_df[filtered_df["at_risk"] == 1]
        elif at_risk_filter == "Not At Risk (G3 >= 10)":
            filtered_df = filtered_df[filtered_df["at_risk"] == 0]

        st.caption(f"Displaying **{len(filtered_df)}** of {len(df)} records.")
        st.dataframe(filtered_df, use_container_width=True, height=350)

        # Download button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="student_performance_filtered.csv",
            mime="text/csv"
        )

        st.markdown("### 📖 Feature Dictionary")
        dict_data = {
            "Feature": ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu/Fedu", "studytime", "failures", "absences", "G1/G2", "G3"],
            "Description": [
                "Student's school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)",
                "Student's biological sex ('F' - female, 'M' - male)",
                "Student's age in years (numeric: 15 to 22)",
                "Home address environment ('U' - urban, 'R' - rural)",
                "Family size ('LE3' <= 3 members, 'GT3' > 3 members)",
                "Parental cohabitation status ('T' - living together, 'A' - apart)",
                "Mother / Father education level (0: none to 4: higher education)",
                "Weekly study time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)",
                "Number of past academic failures (0 to 3)",
                "Number of recorded school absences (0 to 93)",
                "First and second period interim examination marks (0 to 20)",
                "Final academic grade (target variable, 0 to 20 scale)"
            ]
        }
        st.table(pd.DataFrame(dict_data))


# ==============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical distributions, correlation structures, and academic behavioral trends</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Grade Distributions", "🔗 Correlations & Heatmap", "📚 Behavioral & Demographic Factors"])

    with tab1:
        st.markdown("#### Distribution of Academic Grades (G1, G2, G3)")
        fig_grades = go.Figure()
        fig_grades.add_trace(go.Histogram(x=df["G1"], name="G1 (First Period)", opacity=0.7, marker_color="#2B5C8F"))
        fig_grades.add_trace(go.Histogram(x=df["G2"], name="G2 (Second Period)", opacity=0.7, marker_color="#3A7EBF"))
        fig_grades.add_trace(go.Histogram(x=df["G3"], name="G3 (Final Grade)", opacity=0.8, marker_color="#E67E22"))
        fig_grades.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Pass Threshold (10)")
        fig_grades.update_layout(barmode="overlay", title="Comparison of Grade Distributions across Periods", xaxis_title="Score (0-20)", yaxis_title="Student Count")
        st.plotly_chart(fig_grades, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Final Grade (G3)", f"{df['G3'].mean():.2f}")
        c2.metric("Median Final Grade", f"{df['G3'].median():.1f}")
        c3.metric("At-Risk Student Ratio", f"{(df['at_risk'].mean()*100):.1f}%")

    with tab2:
        st.markdown("#### Correlation Matrix of Numeric Attributes")
        numeric_cols = ["age", "Medu", "Fedu", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Heatmap of Academic & Behavioral Variables"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("#### Behavioral Relationships with Final Performance")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_study = px.box(
                df, x="studytime", y="G3", color="at_risk",
                labels={"studytime": "Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", "G3": "Final Grade"},
                title="Weekly Study Time vs Final Grade",
                color_discrete_map={0: "#27AE60", 1: "#E74C3C"}
            )
            st.plotly_chart(fig_study, use_container_width=True)
        with col_b:
            fig_abs = px.scatter(
                df, x="absences", y="G3", color="at_risk", trendline="ols",
                labels={"absences": "Number of Absences", "G3": "Final Grade"},
                title="School Absences vs Final Grade with OLS Trend",
                color_discrete_map={0: "#27AE60", 1: "#E74C3C"}
            )
            st.plotly_chart(fig_abs, use_container_width=True)


# ==============================================================================
# PAGE 4: STUDENT PERFORMANCE PREDICTION
# ==============================================================================
elif page == "Student Performance Prediction":
    st.markdown('<div class="main-title">Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate individualized final grade forecasts with statistical uncertainty intervals</div>', unsafe_allow_html=True)

    exp_mode = st.radio(
        "Select Prediction Experiment Mode:",
        ["Experiment A: Early Prediction (Pre-exam factors only)", "Experiment B: Mid-Term Prediction (Includes G1 & G2)"],
        horizontal=True
    )
    exp_code = "A" if "Experiment A" in exp_mode else "B"

    st.markdown("#### 📝 Student Profile Input")
    with st.form("student_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            school = st.selectbox("School", ["GP", "MS"], help="GP: Gabriel Pereira, MS: Mousinho da Silveira")
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U", "R"], help="U: Urban, R: Rural")
            famsize = st.selectbox("Family Size", ["GT3", "LE3"], help="GT3: >3, LE3: <=3")
            pstatus = st.selectbox("Parent Cohabitation", ["T", "A"], help="T: Together, A: Apart")

        with col2:
            medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=3, help="0: None, 4: Higher")
            fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], index=3)
            mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], index=2)
            fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], index=4)
            reason = st.selectbox("School Choice Reason", ["course", "home", "reputation", "other"])
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])

        with col3:
            studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], index=1, help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
            failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
            absences = st.slider("Recorded Absences", 0, 50, 4)
            schoolsup = st.selectbox("Extra School Support", ["no", "yes"])
            famsup = st.selectbox("Family Support", ["yes", "no"])
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])

        if exp_code == "B":
            st.markdown("##### 📌 Interim Evaluation Marks (Required for Mid-Term Prediction)")
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                g1_val = st.slider("First Period Grade (G1: 0-20)", 0, 20, 11)
            with c_g2:
                g2_val = st.slider("Second Period Grade (G2: 0-20)", 0, 20, 11)
        else:
            g1_val, g2_val = 0, 0

        submit_btn = st.form_submit_button("🔮 Predict Academic Performance", use_container_width=True)

    if submit_btn:
        student_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": 1, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": "no",
            "activities": "yes", "nursery": "yes", "higher": higher, "internet": "yes",
            "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1,
            "health": 3, "absences": absences
        }
        if exp_code == "B":
            student_data["G1"] = g1_val
            student_data["G2"] = g2_val

        with st.spinner("Executing model inference and SHAP attribution..."):
            pred_results = predict_single_student(student_data, experiment=exp_code)

        st.markdown("### 📊 Prediction Results & Confidence Interval")
        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Final Grade (G3)</div>
                <div class="metric-value">{pred_results['predicted_grade']} / 20</div>
            </div>
            """, unsafe_allow_html=True)
        with res2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">95% Confidence Interval</div>
                <div class="metric-value">[{pred_results['ci_lower']}, {pred_results['ci_upper']}]</div>
            </div>
            """, unsafe_allow_html=True)
        with res3:
            badge_class = "badge-risk" if pred_results["is_at_risk"] else "badge-pass"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">At-Risk Classification</div>
                <div class="metric-value"><span class="{badge_class}">{pred_results['risk_level']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔍 Individualized SHAP Feature Contributions")
        st.info("The waterfall plot below illustrates how each feature pushed the predicted grade higher (positive attribution) or lower (negative attribution) relative to the baseline dataset average.")

        try:
            fig_waterfall = plot_local_waterfall(pred_results["shap_explanation"], max_display=10)
            st.pyplot(fig_waterfall)
        except Exception as e:
            st.warning(f"Could not render waterfall plot directly: {e}")

        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("##### 🟢 Top Positive Grade Contributors")
            st.dataframe(pred_results["top_positive_features"], use_container_width=True)
        with c_neg:
            st.markdown("##### 🔴 Top Negative Risk Contributors")
            st.dataframe(pred_results["top_negative_features"], use_container_width=True)


# ==============================================================================
# PAGE 5: AT-RISK ANALYSIS & GUIDANCE
# ==============================================================================
elif page == "At-Risk Analysis & Guidance":
    st.markdown('<div class="main-title">At-Risk Analysis & Academic Intervention Guidance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated screening with compassionate, non-judgmental remediation advice</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Empirical Risk Stratification")
    st.markdown("""
    Students are classified as **At-Risk** if their predicted or actual final grade is below **10/20** ($G3 < 10$).
    We stratify intervention severity into three operational tiers:
    - **High Risk ($P \\ge 0.65$)**: Urgent multidimensional intervention required.
    - **Moderate Risk ($0.35 \\le P < 0.65$)**: Targeted academic monitoring and milestone checks.
    - **Low Risk ($P < 0.35$)**: Progressing well; enrichment and positive reinforcement recommended.
    """)

    sample_student = {
        "school": "GP", "sex": "M", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course",
        "guardian": "mother", "traveltime": 2, "studytime": 1, "failures": 1, "schoolsup": "no",
        "famsup": "yes", "paid": "no", "activities": "no", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no", "famrel": 3, "freetime": 4, "goout": 4,
        "Dalc": 2, "Walc": 3, "health": 3, "absences": 12
    }

    st.markdown("#### Sample Evaluated Case:")
    res_sample = predict_single_student(sample_student, experiment="A")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Predicted Grade**: `{res_sample['predicted_grade']} / 20`")
        st.markdown(f"**At-Risk Probability**: `{res_sample['at_risk_probability']*100:.1f}%`")
        st.markdown(f"**Risk Level**: `{res_sample['risk_level']}`")
    with c2:
        st.markdown(f"**Recorded Absences**: `12 days`")
        st.markdown(f"**Weekly Study Time**: `< 2 hours/week`")
        st.markdown(f"**Prior Failures**: `1 failure`")

    st.markdown("### 📋 Recommended Pedagogical Interventions")
    for rec in res_sample["recommendations"]:
        with st.expander(f"📍 {rec['category']} — [{rec['priority']} Priority]", expanded=True):
            st.markdown(f"**Observation**: {rec['observation']}")
            st.markdown(f"**Actionable Suggestion**: {rec['suggestion']}")

    st.caption("⚠️ Disclaimer: Recommendations are advisory decision-support prompts for educators, not automated verdicts or psychological diagnostics.")


# ==============================================================================
# PAGE 6: EXPLAINABLE AI (SHAP)
# ==============================================================================
elif page == "Explainable AI (SHAP)":
    st.markdown('<div class="main-title">Explainable AI (SHAP) Global Interpretability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Model-agnostic game-theoretic feature attribution across experiments</div>', unsafe_allow_html=True)

    exp_choice = st.selectbox("Choose Experiment Analysis", ["Experiment A: Early Prediction (Pre-Exam)", "Experiment B: Mid-Term Prediction (With G1 & G2)"])

    if "Experiment A" in exp_choice:
        st.markdown("### 🐝 SHAP Beeswarm Summary Plot — Experiment A (Early Prediction)")
        st.markdown("""
        In the beeswarm plot, each dot corresponds to an individual student in the test set.
        - **Red dots** represent higher values of that feature; **Blue dots** represent lower values.
        - Dots to the **right** of the centerline increase final grade ($G3$), while dots to the **left** decrease it.
        """)
        fig_a = FIGURES_DIR / "07_shap_summary_exp_a.png"
        if fig_a.exists():
            st.image(str(fig_a), use_container_width=True)

        st.markdown("### 📊 Mean Absolute SHAP Value (Global Feature Importance)")
        bar_a = FIGURES_DIR / "08_shap_bar_exp_a.png"
        if bar_a.exists():
            st.image(str(bar_a), use_container_width=True)

    else:
        st.markdown("### 🐝 SHAP Beeswarm Summary Plot — Experiment B (Mid-Term Prediction)")
        fig_b = FIGURES_DIR / "09_shap_summary_exp_b.png"
        if fig_b.exists():
            st.image(str(fig_b), use_container_width=True)

        st.markdown("### 📊 Mean Absolute SHAP Value — Experiment B")
        bar_b = FIGURES_DIR / "10_shap_bar_exp_b.png"
        if bar_b.exists():
            st.image(str(bar_b), use_container_width=True)


# ==============================================================================
# PAGE 7: MODEL COMPARISON & BENCHMARKS
# ==============================================================================
elif page == "Model Comparison & Benchmarks":
    st.markdown('<div class="main-title">Model Comparison & Experimental Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Systematic performance evaluation across Regression and Classification algorithms</div>', unsafe_allow_html=True)

    st.markdown("### 📈 Regression Model Evaluation Metrics")
    st.dataframe(reg_metrics, use_container_width=True)

    st.markdown("### 🎯 Classification Model Evaluation Metrics (At-Risk Detection)")
    st.dataframe(clf_metrics, use_container_width=True)

    st.markdown("### 🔬 5-Fold Cross Validation Benchmarks")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Regression CV ($R^2$ & MAE)")
        st.dataframe(cv_reg_metrics, use_container_width=True)
    with col2:
        st.markdown("##### Classification Stratified CV (F1 & ROC-AUC)")
        st.dataframe(cv_clf_metrics, use_container_width=True)

    # Interactive Bar Chart of Metrics
    if not reg_metrics.empty:
        fig_bar = px.bar(
            reg_metrics, x="Model", y="R2", color="Experiment", barmode="group",
            title="Regression R² Score Comparison: Experiment A vs Experiment B",
            color_discrete_sequence=["#1F77B4", "#2CA02C"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PAGE 8: BATCH CSV PREDICTION
# ==============================================================================
elif page == "Batch CSV Prediction":
    st.markdown('<div class="main-title">Batch Student Prediction via CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload institutional cohort records for automated batch evaluation and risk labeling</div>', unsafe_allow_html=True)

    batch_exp = st.radio("Select Prediction Mode for Batch Evaluation:", ["Experiment A (Early)", "Experiment B (Mid-Term)"], horizontal=True)
    b_code = "A" if "Experiment A" in batch_exp else "B"

    # Template download
    template_str = generate_sample_template_csv(b_code)
    st.download_button(
        label=f"📄 Download Sample CSV Template ({b_code})",
        data=template_str,
        file_name=f"sample_student_batch_{b_code}.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Student Dataset (.CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read()
            results_df, summary = predict_batch_csv(content, experiment=b_code)

            st.success(f"Successfully processed {summary['total_records']} student records!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Predicted G3", f"{summary['mean_predicted_g3']:.2f}")
            c2.metric("Total Flagged At-Risk", f"{summary['at_risk_count']}")
            c3.metric("At-Risk Proportion", f"{summary['at_risk_rate']*100:.1f}%")

            st.markdown("### 📋 Prediction Results Preview")
            st.dataframe(results_df, use_container_width=True)

            res_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Enriched Prediction Results CSV",
                data=res_csv,
                file_name="batch_predictions_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error during batch prediction: {str(e)}")


# ==============================================================================
# PAGE 9: ABOUT PROJECT & ETHICS
# ==============================================================================
elif page == "About Project & Ethics":
    st.markdown('<div class="main-title">About Project & Responsible AI Statement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Academic provenance, ethical considerations, and methodological transparency</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ Academic Context
    - **Project**: Minor Research Project (B.Tech CSE)
    - **Title**: *Explainable AI-Based Student Academic Performance Prediction and Early Intervention System*
    - **Lead Developer**: Mukesh Pramanik
    - **Core Tech Stack**: Python 3.14, Scikit-Learn, SHAP, Streamlit, Pandas, NumPy, Plotly

    ### 🛡️ Responsible AI & Ethical Principles
    1. **Strictly Decision-Support**: This system is designed as an advisory tool for educators and academic counseling committees. It must **never** be used to automate penalties, deny admissions, or assign punitive academic restrictions.
    2. **Demographic Safeguards**: Subgroup audits ensure predictions are evaluated for demographic balance. Demographic attributes (such as sex or address) must never serve as standalone criteria for adverse administrative interventions.
    3. **Transparency over Opacity**: Complex ensemble predictions are paired with SHAP explainability so instructors can evaluate the rationale behind any model output.
    4. **Contextual Limitations**: The underlying dataset reflects secondary school mathematics curricula from two Portuguese institutions. Direct transfer to university-level engineering courses in other geographic regions requires local recalibration.
    """)
