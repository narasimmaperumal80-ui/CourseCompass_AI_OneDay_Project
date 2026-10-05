from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESOURCES = pd.read_csv(ROOT / "data" / "resources.csv")

def get_resources(course_name):
    return RESOURCES[RESOURCES["course_name"] == course_name].copy()

def get_study_plan(course_name, available_hours):
    rows = get_resources(course_name)
    base = int(rows["suggested_hours"].sum()) if not rows.empty else 8
    if available_hours <= 0:
        return 0
    return round(min(float(available_hours), base), 1)
