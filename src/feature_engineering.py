"""
Feature engineering functions for academic and behavioral indicators.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-relevant composite features from raw attributes."""
    df_feat = df.copy()

    # 1. Parental education mean index
    if "Medu" in df_feat.columns and "Fedu" in df_feat.columns:
        df_feat["parent_edu_avg"] = (df_feat["Medu"] + df_feat["Fedu"]) / 2.0

    # 2. Total alcohol consumption index (weighted average of workday and weekend)
    if "Dalc" in df_feat.columns and "Walc" in df_feat.columns:
        df_feat["total_alc"] = (df_feat["Dalc"] * 5 + df_feat["Walc"] * 2) / 7.0

    # 3. Study to travel time ratio
    if "studytime" in df_feat.columns and "traveltime" in df_feat.columns:
        df_feat["study_travel_ratio"] = df_feat["studytime"] / (df_feat["traveltime"] + 0.1)

    # 4. Academic support composite indicator
    if "schoolsup" in df_feat.columns and "famsup" in df_feat.columns:
        df_feat["academic_support_any"] = (
            (df_feat["schoolsup"] == "yes") | (df_feat["famsup"] == "yes")
        ).astype(int)

    # 5. At-risk classification target based on final grade threshold (G3 < 10)
    if "G3" in df_feat.columns:
        df_feat["at_risk"] = (df_feat["G3"] < 10).astype(int)

    return df_feat
