"""
Train the CourseCompass future-demand model.

IMPORTANT: demand_data.csv is SYNTHETIC DEMO/TREND DATA.
It demonstrates the ML pipeline. Replace it with a verified real dataset
for any real-world labor-market claim.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

courses = pd.read_csv(DATA / "courses.csv")
rng = np.random.default_rng(7)

# Synthetic historical trend data: 2018-2025.
base = {
    "Machine Learning": (42000, 0.17),
    "Data Analytics": (50000, 0.13),
    "Cloud Computing": (46000, 0.15),
    "Cybersecurity": (43000, 0.16),
    "Web Development": (60000, 0.08),
    "Mobile App Development": (39000, 0.09),
    "Database Engineering": (34000, 0.07),
    "DevOps Engineering": (36000, 0.14),
    "UI/UX Design": (30000, 0.09),
    "Business Intelligence": (37000, 0.12),
}

rows = []
for course, (start, growth) in base.items():
    popularity = rng.uniform(0.45, 0.95)
    for year in range(2018, 2026):
        years = year - 2018
        jobs = start * ((1 + growth) ** years) * rng.normal(1, 0.055)
        industry = np.clip(50 + growth * 180 + years * 1.8 + rng.normal(0, 3), 0, 100)
        salary_index = 70 + growth * 120 + years * 2 + rng.normal(0, 2)
        rows.append({
            "year": year,
            "course_name": course,
            "job_postings": int(max(5000, jobs)),
            "industry_demand": round(float(industry), 2),
            "growth_rate": round(float(growth * 100 + rng.normal(0, 1.0)), 2),
            "salary_index": round(float(salary_index), 2),
            "course_popularity": round(float(popularity + rng.normal(0, 0.03)), 3)
        })

df = pd.DataFrame(rows)
df.to_csv(DATA / "demand_data.csv", index=False)

features = ["year", "course_name", "industry_demand", "growth_rate", "salary_index", "course_popularity"]
target = "job_postings"

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["course_name"]),
    ("num", StandardScaler(), ["year","industry_demand","growth_rate","salary_index","course_popularity"])
])

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3)
}

results = []
best_name, best_model, best_r2 = None, None, -float("inf")

for name, reg in models.items():
    pipe = Pipeline([("preprocess", pre), ("model", reg)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    }
    results.append(metrics)
    if metrics["R2"] > best_r2:
        best_name, best_model, best_r2 = name, pipe, metrics["R2"]

pd.DataFrame(results).sort_values("R2", ascending=False).to_csv(
    MODELS / "demand_model_comparison.csv", index=False
)

joblib.dump(
    {"model": best_model, "features": features, "model_name": best_name},
    MODELS / "demand_model.pkl"
)

print("\nDemand model training complete.")
print(pd.DataFrame(results).sort_values("R2", ascending=False).to_string(index=False))
print(f"\nSelected model: {best_name}")
