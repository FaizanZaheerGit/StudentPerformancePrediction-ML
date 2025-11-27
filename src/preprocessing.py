import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np


def prepare_features(df: pd.DataFrame, target_col: str = "Class"):
    data = df.copy()

    grade_map = {f"G-{i:02d}": i for i in range(1, 13)}
    if "GradeID" in data.columns:
        data["GradeID"] = (
            data["GradeID"]
            .astype(str)
            .map(grade_map)
            .fillna(-1)
        )

    if "StudentAbsenceDays" in data.columns:
        data["StudentAbsenceDays"] = data["StudentAbsenceDays"].map({
            "Under-7": 1,
            "Above-7": 0
        }).fillna(0)

    preferred = ["raisedhands", "VisITedResources", "Discussion", "StudentAbsenceDays"]
    numeric_features = [c for c in preferred if c in data.columns]

    if len(numeric_features) < 4:
        numeric_candidates = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_candidates = [c for c in numeric_candidates if c != target_col]
        if len(numeric_candidates) < 4:
            raise ValueError(f"Not enough numeric features found. Candidates: {numeric_candidates}")
        numeric_features = numeric_candidates[:4]

    imputer = SimpleImputer(strategy="median")
    data[numeric_features] = imputer.fit_transform(data[numeric_features])

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    le_target = LabelEncoder()
    y = le_target.fit_transform(data[target_col])

    X = data[numeric_features].copy()

    return X, y, numeric_features, le_target
