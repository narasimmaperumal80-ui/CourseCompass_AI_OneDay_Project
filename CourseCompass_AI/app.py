import streamlit as st
import pandas as pd
import plotly.express as px
from recommendation import recommend
from resources import get_resources, get_study_plan
from demand import predict_demand

st.set_page_config(
    page_title="CourseCompass AI",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 18px; opacity: .75; margin-bottom: 20px;}
.card {padding: 18px; border-radius: 14px; border: 1px solid #ddd; margin-bottom: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 CourseCompass AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent Course Recommendation and Academic Success Prediction Platform</div>',
    unsafe_allow_html=True
)

st.info(
    "Prototype note: the bundled student and labor-market datasets are synthetic/demo data. "
    "Demand forecasts are estimates from historical/trend patterns and are not guaranteed."
)

with st.sidebar:
    st.header("Student Profile")
    st.write("Enter your academic information.")
    interest = st.selectbox(
        "Primary interest",
        ["AI", "Data", "Cloud", "Security", "Programming", "Design", "Business"]
    )
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    study_hours = st.number_input(
        "Study time per week (hours)", min_value=1.0, max_value=80.0, value=12.0, step=1.0
    )
    st.caption("The course difficulty is evaluated automatically against your academic profile.")

st.subheader("10 Subject Marks")
cols = st.columns(2)
labels = [
    ("math","Mathematics"), ("programming","Programming"), ("statistics","Statistics"),
    ("web_framework","Web Framework"), ("english","English"), ("business","Business"),
    ("ai","Artificial Intelligence"), ("database","Database"), ("networking","Networking"),
    ("cloud","Cloud Computing")
]
marks = {}
for i, (key, label) in enumerate(labels):
    with cols[i % 2]:
        marks[key] = st.slider(label, 0, 100, 70, 1)

if st.button("🚀 Generate CourseCompass Recommendations", type="primary", use_container_width=True):
    result = recommend(marks, cgpa, study_hours, interest)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("Recommended Courses")
    top = result.head(5)

    metric_cols = st.columns(3)
    for i in range(3):
        row = top.iloc[i]
        with metric_cols[i]:
            st.metric(row["course_name"], f'{row["suitability"]:.1f}%')

    st.dataframe(
        top[["course_name","suitability","difficulty","skill_match","base_study_hours","reason"]],
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        top,
        x="suitability",
        y="course_name",
        orientation="h",
        title="Course Suitability Scores"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Learning Resources")
    selected_course = st.selectbox(
        "Choose a recommended course",
        top["course_name"].tolist()
    )
    resources = get_resources(selected_course)
    planned = get_study_plan(selected_course, study_hours)
    st.write(f"Suggested study time for this course: **{planned} hours/week**")

    for _, r in resources.iterrows():
        st.markdown(
            f"- **{r['resource_type']} — {r['title']}** ({r['level']}, ~{r['suggested_hours']}h)  \n"
            f"  {r['url']}"
        )

    st.subheader("Future Demand Estimate")
    demand = predict_demand(selected_course, 2030)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Current demand", f"{demand['current_jobs']:,}")
    d2.metric("2030 estimate", f"{demand['predicted_jobs']:,}")
    d3.metric("Estimated growth", f"{demand['growth_percent']}%")
    d4.metric("Demand category", demand["category"])

    hist = demand["history"].copy()
    hist["type"] = "Historical/demo"
    future = pd.DataFrame({
        "year": [2030],
        "job_postings": [demand["predicted_jobs"]],
        "type": ["ML estimate"]
    })
    chart_df = pd.concat([
        hist[["year","job_postings","type"]],
        future
    ], ignore_index=True)

    fig2 = px.line(
        chart_df, x="year", y="job_postings", markers=True,
        color="type", title=f"{selected_course}: Historical Trend and 2030 Estimate"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "Demand is a model estimate based on the included demo trend dataset. "
        "It should not be interpreted as a guaranteed prediction of jobs, salary, or career success."
    )

    st.subheader("How the recommendation works")
    st.write(
        "The system transforms your 10 marks, CGPA, study time, interest, course difficulty, "
        "and course-specific skill alignment into features. A trained classification model "
        "then estimates the probability that each course is a suitable match. The dashboard "
        "sorts courses by that probability and shows the strongest matches."
    )

else:
    st.write("Fill in the student profile and marks, then click **Generate CourseCompass Recommendations**.")
