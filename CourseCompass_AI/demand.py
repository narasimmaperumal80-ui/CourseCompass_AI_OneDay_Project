from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / "data" / "demand_data.csv")
BUNDLE = joblib.load(ROOT / "models" / "demand_model.pkl")

def predict_demand(course_name, target_year=2030):
    course_hist = DATA[DATA["course_name"] == course_name].sort_values("year")
    if course_hist.empty:
        return None

    last = course_hist.iloc[-1]
    row = pd.DataFrame([{
        "year": target_year,
        "course_name": course_name,
        "industry_demand": last["industry_demand"],
        "growth_rate": last["growth_rate"],
        "salary_index": last["salary_index"],
        "course_popularity": last["course_popularity"]
    }])

    predicted_jobs = float(max(0, BUNDLE["model"].predict(row[BUNDLE["features"]])[0]))
    current_jobs = float(last["job_postings"])
    growth = (predicted_jobs - current_jobs) / current_jobs * 100 if current_jobs else 0

    if growth >= 35:
        category = "Very High"
    elif growth >= 15:
        category = "High"
    elif growth >= 5:
        category = "Medium"
    else:
        category = "Low"

    return {
        "current_jobs": int(current_jobs),
        "predicted_jobs": int(predicted_jobs),
        "growth_percent": round(growth, 1),
        "category": category,
        "history": course_hist
    }
