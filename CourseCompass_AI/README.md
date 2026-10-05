# CourseCompass AI

## Intelligent Course Recommendation and Academic Success Prediction Platform

A medium-level, one-day prototype for recommending courses from a student's academic profile and estimating future course/skill demand.

### Important data note
The included `student_data.csv` and `demand_data.csv` are generated synthetic/demo datasets. They are intended to demonstrate the complete ML pipeline. They must not be described in the academic report as real student or real labor-market observations.

### Features
- 10 subject marks
- CGPA
- Weekly study time
- Primary interest
- Course-specific skill alignment
- ML-based suitability classification
- Model comparison: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Structured learning resources
- Future demand regression
- Demand model comparison
- Streamlit dashboard
- Charts and explanations

## 1. Install Python
Use Python 3.10+.

## 2. Open the project folder
```bash
cd CourseCompass_AI
```

## 3. Create a virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

## 5. Train the models
Run:
```bash
python train_recommendation.py
python train_demand.py
```

This creates:
- `data/student_data.csv`
- `data/demand_data.csv`
- `models/recommendation_model.pkl`
- `models/demand_model.pkl`
- model comparison CSV files

## 6. Start the dashboard
```bash
streamlit run app.py
```

The browser should open the CourseCompass AI dashboard.

## 7. If the browser does not open
Copy the Local URL shown in the terminal, usually:
`http://localhost:8501`

## 8. Recommended screenshots for the report
Capture:
1. Project folder
2. Dataset preview
3. Recommendation model comparison
4. Demand model comparison
5. Streamlit input form
6. Top course recommendations
7. Suitability chart
8. Learning resource section
9. Demand forecast chart
10. Final dashboard

## 9. Viva explanation
### What is the recommendation problem?
For each student-course pair, the model predicts whether the course is a suitable match.

### Why is it classification?
The training target is `suitable`: 0 = not suitable, 1 = suitable. The probability of class 1 is converted into a suitability percentage.

### Why use multiple models?
Different algorithms learn patterns differently. The project compares several models and selects the best model using F1 score.

### Why F1 score?
A course recommendation system should balance false positives and false negatives. F1 combines precision and recall.

### What is the demand model?
It is a regression model that estimates future job-posting demand from year, course, industry demand, growth rate, salary index, and popularity.

### Is the demand prediction guaranteed?
No. It is an estimate based on the available trend data. With synthetic demo data it is only a technical demonstration.

### How can the project become real?
Replace the synthetic datasets with verified historical student outcomes and verified labor-market data, then retrain and re-evaluate the models.

## 10. Common errors
### `ModuleNotFoundError`
Run:
```bash
pip install -r requirements.txt
```

### `FileNotFoundError: recommendation_model.pkl`
Run:
```bash
python train_recommendation.py
```

### `FileNotFoundError: demand_model.pkl`
Run:
```bash
python train_demand.py
```

### Streamlit command not found
Run:
```bash
python -m streamlit run app.py
```

### PowerShell activation error
Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then activate the environment again.
