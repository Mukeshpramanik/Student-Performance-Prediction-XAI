"""
Exploratory Data Analysis (EDA) module generating figures and statistical tables.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import FIGURES_DIR, TABLES_DIR, AT_RISK_THRESHOLD
from src.data_loader import load_raw_data
from src.feature_engineering import add_engineered_features

# Set research figure styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Dict[str, Path]:
    """Generate and save publication-quality EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {}

    # 1. Final Grade (G3) and Period Grades Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, col in enumerate(["G1", "G2", "G3"]):
        sns.histplot(df[col], kde=True, ax=axes[i], color=["#2b5c8f", "#3a7ebf", "#d95f02"][i], bins=15)
        axes[i].axvline(AT_RISK_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label=f"At-Risk Threshold ({AT_RISK_THRESHOLD})")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(f"{col} Score (0-20)")
        axes[i].set_ylabel("Student Count" if i == 0 else "")
        axes[i].legend(loc="upper left")
    plt.tight_layout()
    p1 = output_dir / "01_grade_distribution.png"
    plt.savefig(p1)
    plt.close()
    figure_paths["grade_distribution"] = p1

    # 2. Correlation Heatmap of Numeric Features
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax,
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix of Numeric Attributes & Derived Features", pad=12)
    p2 = output_dir / "02_correlation_heatmap.png"
    plt.savefig(p2)
    plt.close()
    figure_paths["correlation_heatmap"] = p2

    # 3. Study Time vs Final Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    study_labels = {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}
    df_plot = df.copy()
    df_plot["studytime_label"] = df_plot["studytime"].map(study_labels)
    sns.boxplot(x="studytime_label", y="G3", data=df_plot, palette="Blues_d", ax=ax)
    sns.stripplot(x="studytime_label", y="G3", data=df_plot, color="black", alpha=0.3, jitter=0.2, ax=ax)
    ax.axhline(AT_RISK_THRESHOLD, color="red", linestyle="--", label=f"At-Risk Threshold ({AT_RISK_THRESHOLD})")
    ax.set_title("Final Academic Grade (G3) Across Weekly Study Time Categories")
    ax.set_xlabel("Weekly Study Time")
    ax.set_ylabel("Final Grade (G3: 0-20)")
    ax.legend()
    p3 = output_dir / "03_studytime_vs_grades.png"
    plt.savefig(p3)
    plt.close()
    figure_paths["studytime_vs_grades"] = p3

    # 4. Absences vs Final Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x="absences", y="G3", data=df, scatter_kws={"alpha": 0.5, "color": "#1f77b4"},
                line_kws={"color": "darkred", "linewidth": 2}, ax=ax)
    ax.axhline(AT_RISK_THRESHOLD, color="grey", linestyle=":")
    ax.set_title("Relationship Between School Absences and Final Grade (G3)")
    ax.set_xlabel("Total Absences")
    ax.set_ylabel("Final Grade (G3: 0-20)")
    p4 = output_dir / "04_absences_vs_grades.png"
    plt.savefig(p4)
    plt.close()
    figure_paths["absences_vs_grades"] = p4

    # 5. Previous Exam Grades (G1, G2) vs Final Grade (G3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x="G1", y="G3", hue="at_risk", palette={0: "#2ca02c", 1: "#d62728"}, data=df, ax=axes[0], alpha=0.8)
    axes[0].plot([0, 20], [0, 20], "k--", alpha=0.5, label="y = x")
    axes[0].set_title("First Period Grade (G1) vs Final Grade (G3)")
    axes[0].set_xlabel("G1 Score")
    axes[0].set_ylabel("G3 Score")
    axes[0].legend(title="At Risk", labels=["Not At Risk (>=10)", "At Risk (<10)"])

    sns.scatterplot(x="G2", y="G3", hue="at_risk", palette={0: "#2ca02c", 1: "#d62728"}, data=df, ax=axes[1], alpha=0.8)
    axes[1].plot([0, 20], [0, 20], "k--", alpha=0.5, label="y = x")
    axes[1].set_title("Second Period Grade (G2) vs Final Grade (G3)")
    axes[1].set_xlabel("G2 Score")
    axes[1].set_ylabel("G3 Score")
    axes[1].legend(title="At Risk", labels=["Not At Risk (>=10)", "At Risk (<10)"])

    p5 = output_dir / "05_previous_grades_relationship.png"
    plt.savefig(p5)
    plt.close()
    figure_paths["previous_grades"] = p5

    # 6. Past Class Failures Impact
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="failures", hue="at_risk", data=df, palette=["#2b5c8f", "#d95f02"], ax=ax)
    ax.set_title("Number of Past Academic Failures vs Academic At-Risk Status")
    ax.set_xlabel("Past Failures Count")
    ax.set_ylabel("Number of Students")
    ax.legend(title="Status", labels=["Not At Risk (G3 >= 10)", "At Risk (G3 < 10)"])
    p6 = output_dir / "06_failures_impact.png"
    plt.savefig(p6)
    plt.close()
    figure_paths["failures_impact"] = p6

    return figure_paths


def generate_eda_tables(df: pd.DataFrame, output_dir: Path = TABLES_DIR) -> Dict[str, Path]:
    """Generate and save statistical summaries as CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {}

    # 1. Summary Statistics
    desc = df.describe().T
    desc_path = output_dir / "summary_statistics.csv"
    desc.to_csv(desc_path)
    table_paths["summary_statistics"] = desc_path

    # 2. Missing Values & Types
    meta_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(d) for d in df.dtypes],
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percent": (df.isnull().sum().values / len(df)) * 100,
        "Unique_Values": [df[col].nunique() for col in df.columns]
    })
    meta_path = output_dir / "feature_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    table_paths["feature_metadata"] = meta_path

    # 3. Subgroup summary across sex and school
    subgroups = df.groupby(["school", "sex"]).agg(
        student_count=("G3", "count"),
        g3_mean=("G3", "mean"),
        g3_median=("G3", "median"),
        g3_std=("G3", "std"),
        at_risk_count=("at_risk", "sum"),
        at_risk_rate=("at_risk", "mean")
    ).reset_index()
    sub_path = output_dir / "subgroup_summary.csv"
    subgroups.to_csv(sub_path, index=False)
    table_paths["subgroup_summary"] = sub_path

    return table_paths


def run_full_eda() -> Dict[str, Any]:
    """Execute complete EDA pipeline and return generated artifacts."""
    raw_df = load_raw_data()
    clean_df = add_engineered_features(raw_df)
    figures = generate_eda_figures(clean_df)
    tables = generate_eda_tables(clean_df)
    return {"figures": figures, "tables": tables, "dataframe": clean_df}


if __name__ == "__main__":
    results = run_full_eda()
    print(f"Generated {len(results['figures'])} figures and {len(results['tables'])} tables successfully.")
