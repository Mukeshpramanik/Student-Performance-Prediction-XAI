"""
Report and presentation generator for the Minor Research Project.
Compiles experimental metrics and generates a 19-slide PowerPoint deck using python-pptx.
"""

from pathlib import Path
import sys
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import REPORTS_DIR, METRICS_DIR, FIGURES_DIR


def create_presentation(output_path: Path = REPORTS_DIR / "Minor_Research_Project_Presentation.pptx"):
    """Generate professional 19-slide presentation matching academic thesis requirements."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Modern 16:9 widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color palette
    NAVY = RGBColor(24, 43, 73)
    SLATE = RGBColor(70, 80, 95)
    DARK_TEXT = RGBColor(30, 30, 30)
    ACCENT_BLUE = RGBColor(41, 128, 185)
    ACCENT_GREEN = RGBColor(39, 174, 96)
    CARD_BG = RGBColor(245, 247, 250)

    slides_content = [
        {
            "title": "Explainable AI-Based Student Academic Performance Prediction\nand Early Intervention System",
            "subtitle": "B.Tech Minor Research Project | Department of Computer Science & Engineering\nStudent: Mukesh Pramanik | Environment: Python, Scikit-Learn, SHAP, Streamlit",
            "is_title_slide": True
        },
        {
            "title": "1. Introduction",
            "bullets": [
                "Student academic achievement is a pivotal metric in educational development and societal progress.",
                "Traditional academic intervention systems are reactive—identifying academic struggle only after exams conclude.",
                "Learning analytics combined with machine learning facilitates proactive early risk detection.",
                "However, complex 'black-box' predictive models face severe adoption barriers in educational governance.",
                "Solution: Integrate Explainable AI (XAI) using SHAP to provide transparent, interpretable, and actionable early intervention recommendations."
            ]
        },
        {
            "title": "2. Problem Statement",
            "bullets": [
                "Lack of Timely Action: Conventional grading frameworks identify failing students after semester completion, preventing timely remediation.",
                "The Black-Box Dilemma: Existing predictive architectures lack interpretability, leaving educators unsure why a student was flagged as 'at-risk'.",
                "Temporal Grade Availability Gap: Early predictions must rely strictly on demographic and behavioral data before exam marks exist.",
                "Absence of Actionable Interventions: Most models output isolated scores without guiding instructors on targeted remedial strategies."
            ]
        },
        {
            "title": "3. Research Objectives",
            "bullets": [
                "RO1: Develop and benchmark regression models (Linear Regression, Decision Trees, Random Forests, Gradient Boosting) for final grade prediction.",
                "RO2: Implement classification architectures to detect students requiring early academic intervention (threshold G3 < 10).",
                "RO3: Systematically isolate the temporal value of interim assessments (Experiment A: Early vs. Experiment B: Mid-Term).",
                "RO4: Integrate SHAP explainability for global feature importance and individualized local student attribution.",
                "RO5: Engineer an interactive, responsive Streamlit decision-support dashboard with automated support recommendations."
            ]
        },
        {
            "title": "4. Literature Review",
            "bullets": [
                "Cortez & Silva (2008): Established benchmark secondary school dataset; demonstrated Random Forest superiority on Portuguese and Math courses.",
                "Lundberg & Lee (2017): Formulated unified framework for SHAP (SHapley Additive exPlanations) rooted in cooperative game theory.",
                "Romero & Ventura (2020): Comprehensive survey on Educational Data Mining (EDM); highlighted need for actionable pedagodical insights.",
                "Conati et al. (2018): Emphasized that pedagogical trust requires transparent student models that articulate reasoning to teachers."
            ]
        },
        {
            "title": "5. Research Gap",
            "bullets": [
                "Rigid Temporal Assumptions: Most prior studies blend mid-term marks (G1, G2) without measuring the performance drop when only early pre-exam indicators exist.",
                "Absence of Local XAI Explanations: Existing educational tools show aggregate feature weights but fail to explain why a specific individual student is flagged.",
                "Disconnection from Pedagogical Practice: Research papers frequently publish raw accuracy scores without delivering a deployable, clinician-grade decision support platform."
            ]
        },
        {
            "title": "6. Dataset & Characteristics",
            "bullets": [
                "Dataset: UCI Machine Learning Repository — Student Performance Dataset (Cortez & Silva, 2008).",
                "Scope: Secondary education Mathematics course (student-mat.csv); 395 student instances across 33 variables.",
                "Integrity: 0 missing values, 0 duplicate records, validated 0-20 grading scale.",
                "Key Dimensions: Demographic (age, sex, address), Academic context (study time, failures, school support), and Behavioral (absences, alcohol consumption).",
                "Target Variables: Final Grade G3 (continuous 0-20) and Binary At-Risk Status (G3 < 10)."
            ]
        },
        {
            "title": "7. Proposed Methodology",
            "bullets": [
                "Data Pipeline: Data loading, domain feature engineering, and robust schema validation.",
                "Leakage-Free Preprocessing: ColumnTransformer with StandardScaler for numerical attributes and OneHotEncoder for categorical attributes fitted strictly on training data.",
                "Two-Stage Experimental Design:",
                "  • Experiment A (Early Prediction): Pre-examination indicators only (excludes G1 & G2).",
                "  • Experiment B (Mid-Term Prediction): Includes interim evaluations (G1 & G2).",
                "Model Evaluation: 5-fold cross-validation and held-out test set (80/20 train/test split)."
            ]
        },
        {
            "title": "8. System Architecture",
            "bullets": [
                "Presentation Layer: Streamlit multi-page dashboard (Home, Explorer, EDA, Prediction, Risk Guidance, SHAP, Benchmarks, Batch, Ethics).",
                "Inference Engine: Scikit-learn pipelines with logistic risk probability estimation and tree-ensemble spread estimates derived from variation across individual trees.",
                "XAI Engine: SHAP TreeExplainer & LinearExplainer computing beeswarm, global bar, and individual waterfall plots.",
                "Advisory Engine: Multi-factor rule-based logic generating empathetic, non-judgmental remediation advice.",
                "Persistence Layer: Joblib model artifacts, structured figures, and CSV evaluation matrices."
            ]
        },
        {
            "title": "9. Exploratory Data Analysis (EDA) Insights",
            "bullets": [
                "Grade Distribution: G3 exhibits a mean of 10.42 (std: 4.58, range 0–20). Bimodal peak with 38 students (9.6%) scoring 0.",
                "Study Time Pattern: Modest positive correlation with final grade (r = 0.098, p = 0.052); students studying <2 hrs/week average 10.05 vs. 11.40 for 5–10 hrs/week.",
                "Past Failures: Strongest pre-exam negative indicator (r = -0.36); at-risk rate rises from 25.0% (0 failures) to 62.7% (1+ failures), a ~2.5x increase.",
                "Absences Impact: Non-linear, right-skewed distribution (median: 4, max: 75; r = 0.034); at-risk rate increases from 30.4% (<=10 absences) to 45.5% (>10 absences)."
            ]
        },
        {
            "title": "10. Machine Learning Models",
            "bullets": [
                "Regression Suite:",
                "  1. Linear Regression (baseline parametric benchmark)",
                "  2. Decision Tree Regressor (non-linear baseline, max_depth=5)",
                "  3. Random Forest Regressor (ensemble bagging, 100 estimators)",
                "  4. Gradient Boosting Regressor (ensemble boosting, learning_rate=0.05)",
                "Classification Suite (At-Risk Threshold G3 < 10):",
                "  1. Logistic Regression (L2 regularization, balanced optimization)",
                "  2. Random Forest Classifier (100 estimators, max_depth=6)"
            ]
        },
        {
            "title": "11. Experimental Results — Regression",
            "bullets": [
                "Experiment A (Early Prediction — Without G1 & G2):",
                "  • Random Forest Regressor achieved MAE = 3.15, RMSE = 3.94, R² = 0.241.",
                "  • Gradient Boosting achieved MAE = 3.21, RMSE = 4.01, R² = 0.215.",
                "  • Confirms that pre-exam demographic/behavioral indicators establish an informative baseline but have intrinsic variance limits.",
                "Experiment B (Mid-Term Prediction — With G1 & G2):",
                "  • Gradient Boosting Regressor achieved MAE = 1.15, RMSE = 1.95, R² = 0.814.",
                "  • Random Forest achieved MAE = 1.20, RMSE = 1.98, R² = 0.808.",
                "  • Inclusion of interim exam scores boosts explained variance by over 57 percentage points!"
            ]
        },
        {
            "title": "12. Experimental Results — Classification",
            "bullets": [
                "Experiment A (Early At-Risk Identification):",
                "  • Logistic Regression achieved Accuracy = 69.6%, ROC-AUC = 0.711, Precision = 57.1%, Recall = 30.8%.",
                "  • Random Forest Classifier achieved Accuracy = 69.6%, ROC-AUC = 0.684.",
                "  • Provides effective screening to prioritize high-risk candidates before midterm exams.",
                "Experiment B (Mid-Term At-Risk Identification):",
                "  • Logistic Regression achieved Accuracy = 91.1%, ROC-AUC = 0.977, F1 = 0.851, Precision = 95.2%.",
                "  • Random Forest Classifier achieved Accuracy = 89.9%, ROC-AUC = 0.961, Recall = 84.6%."
            ]
        },
        {
            "title": "13. Explainable AI Using SHAP",
            "bullets": [
                "Quantified via mean absolute SHAP values (|attribution|), reflecting observational importance within the model.",
                "Global Importance (Exp A — Random Forest): absences (0.9813), failures (0.9097), health (0.1606), goout (0.1485), parent_edu_avg (0.1403).",
                "Global Importance (Exp B — Gradient Boosting): G2 (3.2914), absences (0.8977), G1 (0.2955), reason_home (0.1555), famrel (0.1495).",
                "Local Interpretability (Waterfall Plots): Decomposes individual predictions additively: y_hat = E[f(X)] + sum(phi_i).",
                "Pedagogical Significance: Advisers see verified feature attributions rather than opaque black-box predictions."
            ]
        },
        {
            "title": "14. Subgroup Fairness & Responsible AI",
            "bullets": [
                "Subgroup Audit across Student Sex (Held-out Test Cohort, N=79):",
                "  • Female Subgroup (N=38, 48.10%): MAE = 1.342, RMSE = 2.355, R² = 0.730.",
                "  • Male Subgroup (N=41, 51.90%): MAE = 0.977, RMSE = 1.491, R² = 0.879.",
                "Ethical Interpretation: Differences reflect test sample variance on small groups; formal statistical bias is not claimed.",
                "Responsible AI Protocol: Decisions remain strictly advisory; demographic factors must never trigger punitive academic actions."
            ]
        },
        {
            "title": "15. Streamlit Interactive Dashboard",
            "bullets": [
                "Page 1: Project Overview & Executive Summary Dashboard.",
                "Page 2: Dataset Explorer with dynamic filters, column dictionary, and CSV export.",
                "Page 3: Interactive EDA visualizer with Plotly charts and correlation heatmaps.",
                "Page 4: Student Performance Prediction with tree-ensemble spread estimates and local SHAP waterfall.",
                "Page 5: At-Risk Analysis with risk level categorization and empathetic intervention guidance.",
                "Page 6: XAI Global Explorer with beeswarm plots and feature impact rankings.",
                "Page 7: Model Comparison benchmark tables and CV distributions.",
                "Page 8: Batch CSV Prediction with schema validation and results export.",
                "Page 9: About Project & Ethics Statement."
            ]
        },
        {
            "title": "16. Research Limitations",
            "bullets": [
                "Dataset Geography: Data originates from two Portuguese secondary schools; contextual differences exist when applying to Indian college curricula.",
                "Sample Size: 395 instances is sufficient for robust classical ML and cross-validation, but deep architectures require larger cohorts.",
                "Static Snapshot: Dataset captures a single academic session; continuous longitudinal tracking would enhance trajectory modeling.",
                "Subject Specificity: Analysis is focused on secondary mathematics."
            ]
        },
        {
            "title": "17. Conclusion",
            "bullets": [
                "Successfully built an end-to-end, reproducible, Explainable AI framework for student performance prediction.",
                "Empirically validated that early models (R² = 0.24, AUC = 0.71) offer vital early-screening utility, while midterm models (R² = 0.81, AUC = 0.98) provide high-precision monitoring.",
                "SHAP local waterfall explanations bridge the gap between machine learning models and academic faculty.",
                "Delivered a fully functional 9-page Streamlit dashboard suitable for live institutional demonstrations."
            ]
        },
        {
            "title": "18. Future Scope",
            "bullets": [
                "LMS Integration: Integrate real-time clickstream data from Moodle / Canvas (video watch time, quiz attempts).",
                "Multimodal Learning Analytics: Incorporate natural language processing on student forum discussions and assignments.",
                "Longitudinal Trajectory Models: Implement Recurrent Neural Networks (LSTMs) or Transformers across multi-semester university curricula.",
                "Counterfactual Explanations: Provide automated guidance such as 'If study time increases to 6 hrs/week, predicted grade improves by 2.3 points'."
            ]
        },
        {
            "title": "19. References",
            "bullets": [
                "1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. EUROSIS-ETI, 5-12.",
                "2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS 30), 4765-4774.",
                "3. Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. WIREs Data Mining and Knowledge Discovery, 10(3), e1355.",
                "4. Conati, C., Porayska-Pomsta, K., & Mavrikis, M. (2018). AI in Education needs interpretable machine learning: Lessons from Open Learner Modelling. Proceedings of the Third Annual Workshop on Human Interpretability in Machine Learning (WHI 2018), pp. 21–27. https://doi.org/10.48550/arXiv.1807.00154",
                "5. Barredo Arrieta, A., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges. Information Fusion, 58, 82-115."
            ]
        }
    ]

    for item in slides_content:
        slide = prs.slides.add_slide(blank_layout)

        # Title styling
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = item["title"]
        p.font.name = "Arial"
        p.font.size = Pt(24 if not item.get("is_title_slide") else 28)
        p.font.bold = True
        p.font.color.rgb = NAVY

        if item.get("is_title_slide"):
            p_sub = tf.add_paragraph()
            p_sub.text = "\n" + item["subtitle"]
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(16)
            p_sub.font.color.rgb = SLATE
            continue

        # Content Box
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
        c_tf = content_box.text_frame
        c_tf.word_wrap = True

        is_references = item.get("title") == "19. References"
        for b_idx, bullet in enumerate(item.get("bullets", [])):
            bp = c_tf.add_paragraph() if b_idx > 0 else c_tf.paragraphs[0]
            bp.text = bullet
            bp.font.name = "Arial"
            bp.font.size = Pt(13 if is_references else 16)
            bp.font.color.rgb = DARK_TEXT
            bp.space_after = Pt(8 if is_references else 12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at {output_path} (20 slides).")


if __name__ == "__main__":
    create_presentation()
