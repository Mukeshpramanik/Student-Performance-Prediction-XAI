"""
Data loading and dataset validation utilities.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_FILE, PROCESSED_DATA_FILE, TARGET_COL, AT_RISK_THRESHOLD


def load_raw_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw student mathematics dataset with semicolon delimiter."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}. Run scripts/download_dataset.py first.")
    df = pd.read_csv(file_path, sep=";")
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset structure, check missing values, duplicates, and target bounds."""
    validation_info = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_per_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "target_present": TARGET_COL in df.columns,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    if validation_info["target_present"]:
        target_series = df[TARGET_COL]
        validation_info["target_min"] = float(target_series.min())
        validation_info["target_max"] = float(target_series.max())
        validation_info["target_mean"] = float(target_series.mean())
        validation_info["target_std"] = float(target_series.std())
        validation_info["at_risk_count"] = int((target_series < AT_RISK_THRESHOLD).sum())
        validation_info["not_at_risk_count"] = int((target_series >= AT_RISK_THRESHOLD).sum())
        validation_info["at_risk_ratio"] = float(validation_info["at_risk_count"] / len(df))

    return validation_info


def save_processed_data(df: pd.DataFrame, file_path: Path = PROCESSED_DATA_FILE) -> Path:
    """Persist processed dataset to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path


def load_processed_data(file_path: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    """Load cleaned/processed dataset from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at {file_path}.")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    raw_df = load_raw_data()
    info = validate_dataset(raw_df)
    print(f"Loaded raw dataset with {info['num_rows']} rows and {info['num_columns']} columns.")
    print(f"Missing values: {info['missing_values_total']}, Duplicates: {info['duplicate_rows']}")
    print(f"At-risk count (G3 < {AT_RISK_THRESHOLD}): {info['at_risk_count']} ({info['at_risk_ratio']*100:.2f}%)")
