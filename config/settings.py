"""
Project configuration and directory path management.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "model_metrics"
TABLES_DIR = REPORTS_DIR / "tables"

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "student-mat.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "student_clean.csv"

# Modeling Hyperparameters and Constants
SEED = 42
TARGET_COL = "G3"
AT_RISK_THRESHOLD = 10  # Academic threshold for intervention flag: G3 < 10

# Feature subsets for Experiment A (Early) and Experiment B (Mid-Term)
INTERMEDIATE_GRADE_FEATURES = ["G1", "G2"]
