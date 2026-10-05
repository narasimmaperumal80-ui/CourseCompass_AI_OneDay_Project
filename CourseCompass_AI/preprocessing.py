import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUBJECTS = [
    "math", "programming", "statistics", "web_framework", "english",
    "business", "ai", "database", "networking", "cloud"
]

CATEGORICAL = ["interest"]
NUMERIC = [f"mark_{s}" for s in SUBJECTS] + [
    "cgpa", "study_hours", "skill_match", "course_difficulty"
]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop"
    )

def clean_student_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    if "interest" in df:
        df["interest"] = df["interest"].fillna("AI").astype(str)
    return df

def marks_to_cgpa_estimate(marks):
    return float(np.clip(np.mean(marks) / 10.0, 0, 10))
