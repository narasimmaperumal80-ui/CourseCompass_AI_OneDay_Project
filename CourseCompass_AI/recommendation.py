from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
COURSES = pd.read_csv(ROOT / "data" / "courses.csv")
MODEL_BUNDLE = joblib.load(ROOT / "models" / "recommendation_model.pkl")

SUBJECTS = ["math","programming","statistics","web_framework","english","business","ai","database","networking","cloud"]

def _skill_match(student_marks, cgpa, interest, course):
    weights = course[[f"subject_{s}" for s in SUBJECTS]].to_numpy(dtype=float)
    marks = np.array([student_marks[s] for s in SUBJECTS], dtype=float) / 100
    academic_match = float(np.dot(marks, weights) / max(weights.sum(), 1e-9))
    interest_match = 1.0 if interest == course.primary_interest else (
        0.7 if interest == course.domain else 0.35
    )
    difficulty_match = float(np.clip(
        1 - abs((cgpa / 10 * 5.0) - course.difficulty) / 4.0, 0, 1
    ))
    return float(np.clip(
        0.55 * academic_match + 0.25 * interest_match + 0.20 * difficulty_match,
        0, 1
    ))

def recommend(student_marks, cgpa, study_hours, interest):
    rows = []
    for _, c in COURSES.iterrows():
        skill_match = _skill_match(student_marks, cgpa, interest, c)
        row = {
            **{f"mark_{s}": float(student_marks[s]) for s in SUBJECTS},
            "cgpa": float(cgpa),
            "study_hours": float(study_hours),
            "skill_match": skill_match,
            "course_difficulty": int(c.difficulty)
        }
        rows.append((c, row))

    X = pd.DataFrame([r for _, r in rows])[MODEL_BUNDLE["features"]]
    model = MODEL_BUNDLE["model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X).astype(float)

    results = []
    for (c, row), p in zip(rows, probs):
        suitability = float(np.clip(p * 100, 0, 100))
        reasons = []
        avg = np.mean([student_marks[s] for s in SUBJECTS])
        if row["skill_match"] >= 0.75:
            reasons.append("strong academic/interest alignment")
        elif row["skill_match"] >= 0.55:
            reasons.append("moderate academic/interest alignment")
        else:
            reasons.append("some prerequisite areas may need strengthening")
        if study_hours >= c.base_study_hours:
            reasons.append("available study time matches the typical workload")
        else:
            reasons.append("consider increasing study time for this difficulty")
        if cgpa >= 7.5 and c.difficulty >= 4:
            reasons.append("academic profile supports a challenging course")
        elif c.difficulty <= 2:
            reasons.append("lower difficulty makes the course easier to start")
        results.append({
            "course_name": c.course_name,
            "suitability": suitability,
            "difficulty": int(c.difficulty),
            "domain": c.domain,
            "base_study_hours": int(c.base_study_hours),
            "skill_match": round(row["skill_match"] * 100, 1),
            "reason": "; ".join(reasons)
        })

    return pd.DataFrame(results).sort_values("suitability", ascending=False).reset_index(drop=True)
