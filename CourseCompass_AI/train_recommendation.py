"""
Train the CourseCompass recommendation model.

The generated student_data.csv is SYNTHETIC DEMO DATA.
It is created from plausible academic/course relationships so that a
working prototype can be demonstrated without claiming real student data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

COURSES = pd.read_csv(DATA / "courses.csv")

rng = np.random.default_rng(42)
rows = []
interests = ["AI", "Data", "Cloud", "Security", "Programming", "Design", "Business"]

# 350 synthetic students x 10 courses = 3500 training examples.
for sid in range(350):
    marks = rng.normal(70, 14, 10).clip(35, 98)
    cgpa = np.clip(marks.mean() / 10 + rng.normal(0, 0.25), 4.0, 10.0)
    study_hours = float(rng.uniform(4, 25))
    interest = rng.choice(interests)
    for _, c in COURSES.iterrows():
        subj_weights = np.array([
            c.subject_math, c.subject_programming, c.subject_statistics,
            c.subject_web_framework, c.subject_english, c.subject_business,
            c.subject_ai, c.subject_database, c.subject_networking,
            c.subject_cloud
        ])
        normalized_marks = marks / 100
        academic_match = float(np.dot(normalized_marks, subj_weights) / subj_weights.sum())
        interest_match = 1.0 if interest == c.primary_interest else (
            0.7 if interest == c.domain else 0.35
        )
        time_match = np.clip(study_hours / (c.base_study_hours * 1.2), 0, 1)
        difficulty_match = np.clip(
            1 - abs((cgpa / 10 * 5.0) - c.difficulty) / 4.0, 0, 1
        )
        skill_match = np.clip(
            0.55 * academic_match + 0.25 * interest_match + 0.20 * difficulty_match, 0, 1
        )
        latent = (
            0.48 * academic_match +
            0.20 * interest_match +
            0.14 * time_match +
            0.12 * difficulty_match +
            0.06 * (cgpa / 10)
        )
        score = latent + rng.normal(0, 0.06)
        suitable = int(score >= 0.64)
        rows.append({
            "student_id": sid,
            **{f"mark_{name}": float(m) for name, m in zip(
                ["math","programming","statistics","web_framework","english","business","ai","database","networking","cloud"], marks
            )},
            "cgpa": float(cgpa),
            "study_hours": study_hours,
            "interest": interest,
            "course_name": c.course_name,
            "course_difficulty": int(c.difficulty),
            "skill_match": float(skill_match),
            "suitable": suitable
        })

df = pd.DataFrame(rows)
df.to_csv(DATA / "student_data.csv", index=False)

features = [
    "mark_math","mark_programming","mark_statistics","mark_web_framework","mark_english",
    "mark_business","mark_ai","mark_database","mark_networking","mark_cloud",
    "cgpa","study_hours","skill_match","course_difficulty"
]
X = df[features].copy()
y = df["suitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
best_name, best_model, best_f1 = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    }
    results.append(metrics)
    if metrics["f1"] > best_f1:
        best_name, best_model, best_f1 = name, model, metrics["f1"]

pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
    MODELS / "recommendation_model_comparison.csv", index=False
)

joblib.dump(
    {"model": best_model, "features": features, "model_name": best_name},
    MODELS / "recommendation_model.pkl"
)

print("\nRecommendation model training complete.")
print(pd.DataFrame(results).sort_values("f1", ascending=False).to_string(index=False))
print(f"\nSelected model: {best_name}")
print("\nClassification report:")
print(classification_report(y_test, best_model.predict(X_test), zero_division=0))
