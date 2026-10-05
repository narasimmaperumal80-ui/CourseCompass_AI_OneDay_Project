# CourseCompass AI — 100-Page Academic Report Blueprint

Do not pad the report with repeated text. Use the following structure and add your actual screenshots, generated charts, tables, code excerpts, test evidence, and discussion.

## Suggested page allocation

| Section | Suggested pages |
|---|---:|
| Title, certificate, declaration, acknowledgement | 5 |
| Abstract + keywords | 2 |
| Table of contents/list of figures/tables | 4 |
| Chapter 1: Introduction | 6 |
| Chapter 2: Literature Survey | 8 |
| Chapter 3: Existing and Proposed System | 7 |
| Chapter 4: Requirements and Feasibility | 6 |
| Chapter 5: Methodology | 10 |
| Chapter 6: Dataset and Preprocessing | 9 |
| Chapter 7: ML Models and Training | 10 |
| Chapter 8: Recommendation and Demand Modules | 8 |
| Chapter 9: System Design and Architecture | 6 |
| Chapter 10: Implementation and UI | 8 |
| Chapter 11: Testing | 5 |
| Chapter 12: Results and Discussion | 7 |
| Chapter 13: Limitations and Future Scope | 3 |
| Chapter 14: Conclusion | 2 |
| References | 2 |
| Appendix: code, sample data, test evidence | 10 |
| **Total** | **~100** |

## Abstract
CourseCompass AI is a machine-learning-assisted academic decision-support prototype that analyzes student academic performance, study time, interests, and course characteristics to estimate course suitability. The platform also provides structured learning resources and a separate demand-estimation component. The prototype uses Python, Pandas, NumPy, Scikit-learn, Joblib, Plotly, and Streamlit. Synthetic datasets are used for demonstration and are clearly identified as such. The demand component is an estimate and does not guarantee future employment or salary outcomes.

## Chapter 1 — Introduction
Explain:
- growth of online courses and specialization choices
- difficulty students face when matching academic strengths to courses
- need for personalized guidance
- role of machine learning in decision support
- motivation for CourseCompass AI
- scope and constraints

## Problem Statement
Students may have marks and CGPA but still lack a systematic way to compare courses with their academic profile, available study time, interests, and course difficulty. A prototype is therefore required to provide explainable course suitability estimates, structured resources, and a trend-based future-demand estimate.

## Objectives
1. Build a student profile module.
2. Preprocess numerical and categorical features.
3. Train and compare multiple classification models.
4. Recommend suitable courses with scores.
5. Explain recommendation factors.
6. Provide structured resources.
7. Train a demand regression model.
8. Display outputs through Streamlit.
9. Evaluate the system with measurable metrics.

## Literature Survey
Discuss academic recommender systems, content-based recommendation, classification for educational prediction, learning analytics, labor-market trend analysis, and explainable recommendations. Use real papers/books/web references in the final submitted report rather than inventing citations.

## Existing System
Discuss:
- generic course lists
- non-personalized search
- manual counseling
- fragmented learning resources
- absence of combined academic suitability and trend analysis

## Proposed System
CourseCompass AI combines:
Student profile → preprocessing → feature engineering → ML recommendation → resources
and
Historical/trend data → preprocessing → ML regression → demand estimate.

## Dataset
### Student dataset
Synthetic demonstration records containing:
- 10 subject marks
- CGPA
- study hours
- interest
- course difficulty
- skill-match feature
- suitable target

### Demand dataset
Synthetic demonstration records containing:
- year
- course
- job postings
- industry demand
- growth rate
- salary index
- course popularity

Explicitly label these as synthetic/demo data in the report.

## Preprocessing
Explain:
- missing-value handling
- numeric conversion
- categorical encoding
- scaling
- train/test split
- prevention of data leakage
- reproducibility using fixed random seeds

## Feature Engineering
Explain:
- academic alignment between marks and course subject weights
- interest alignment
- difficulty alignment
- study-time alignment
- combined skill-match score

## Recommendation Models
Compare:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Metrics:
- Accuracy
- Precision
- Recall
- F1 score

Select the model with the strongest F1 score on the held-out test set.

## Demand Models
Compare:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Metrics:
- MAE
- RMSE
- R²

Select the strongest R² model while discussing MAE/RMSE.

## Recommendation Logic
The model predicts suitability probability for every available course. Courses are sorted by probability and displayed with:
- suitability
- difficulty
- skill match
- suggested study time
- explanation

## Resource Module
The resource table contains curated URLs rather than generated URLs. Explain why a structured resource table reduces hallucinated resources.

## Dashboard
Include screenshots of:
- sidebar profile
- 10 subject inputs
- recommendations
- suitability chart
- resources
- demand metrics
- demand trend chart

## Testing
Create a table with:
- test ID
- input
- expected result
- actual result
- status

Example cases:
TC01 valid marks
TC02 CGPA lower boundary
TC03 CGPA upper boundary
TC04 missing/invalid model files
TC05 recommendation generation
TC06 resource retrieval
TC07 demand prediction
TC08 dashboard launch
TC09 unseen interest/category
TC10 extreme study time

## Results
Insert the actual model comparison tables generated by the scripts. Do not invent accuracy/R² values. Discuss why the selected model performed better on this particular demo split.

## Limitations
- synthetic student data
- synthetic demand data
- limited number of course categories
- no real student outcome validation
- labor-market prediction uncertainty
- recommendation scores are decision support, not professional counseling

## Future Scope
- verified institutional student data
- real job-posting datasets
- time-series forecasting
- feedback loop from student outcomes
- explainability with SHAP
- multilingual interface
- college-specific course catalogs
- scholarship/prerequisite integration
- cloud deployment

## Conclusion
Conclude that the prototype demonstrates an end-to-end ML workflow rather than claiming production-grade educational or employment prediction.

## Appendix
Include:
- selected code listings
- sample CSV rows
- model comparison tables
- screenshots
- test results
- commands used
- environment details
