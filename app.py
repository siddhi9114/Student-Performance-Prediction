
import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD TRAINED MODELS ----------------

linear_model = joblib.load("./models/linear_regression_model.pkl")
logistic_model = joblib.load("./models/logistic_regression_model.pkl")
features = joblib.load("./models/features.pkl")


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ---------------- CUSTOM LIGHT THEME ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F3F6FC;
}

/* Main Content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Heading */
h1 {
    color: #243B67 !important;
    font-size: 34px !important;
    font-weight: 800 !important;
    text-align: center;
}

/* Subheadings */
h2, h3 {
    color: #344C7D !important;
    font-weight: 700 !important;
}

/* Normal Text */
p, label, .stMarkdown {
    color: #374151;
}

/* Input Labels */
.stNumberInput label {
    color: #344C7D !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Number Input Boxes */
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #243B67 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Input Focus */
.stNumberInput input:focus {
    border: 1px solid #6366F1 !important;
}

/* Prediction Button */
.stButton > button {
    background: linear-gradient(90deg, #4F46E5, #6366F1);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    transition: 0.3s;
}

/* Button Hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #3730A3, #4F46E5);
    color: white !important;
    border: none;
    transform: translateY(-2px);
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #DCE4F2;
    box-shadow: 0px 3px 10px rgba(30, 50, 90, 0.06);
}

/* Metric Label */
[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 600 !important;
}

/* Metric Value */
[data-testid="stMetricValue"] {
    color: #4F46E5 !important;
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Divider */
hr {
    border-color: #D8E1F0 !important;
}

/* Caption */
.stCaption {
    color: #64748B !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- APPLICATION HEADER ----------------

st.title("🎓 Student Performance Prediction System")

st.markdown(
    """
    <p style="text-align:center; color:#64748B; font-size:16px;">
    Predict student exam scores and academic performance
    using Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# ---------------- STUDENT INPUTS ----------------

st.subheader("📝 Enter Student Details")

st.write("Provide the following academic details to generate a prediction.")

hours_studied = st.number_input(
    "📚 Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_scores = st.number_input(
    "📊 Previous Scores",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

sleep_hours = st.number_input(
    "😴 Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

tutoring_sessions = st.number_input(
    "👨‍🏫 Tutoring Sessions",
    min_value=0,
    max_value=20,
    value=1
)

physical_activity = st.number_input(
    "🏃 Physical Activity (hours per week)",
    min_value=0,
    max_value=24,
    value=3
)


# ---------------- PREDICTION ----------------

st.write("")

if st.button("🔮 Predict Student Performance"):

    input_data = pd.DataFrame([[
        hours_studied,
        attendance,
        previous_scores,
        sleep_hours,
        tutoring_sessions,
        physical_activity
    ]], columns=features)

    # Predict Exam Score
    predicted_score = linear_model.predict(input_data)[0]

    # Predict Performance Category
    predicted_status = logistic_model.predict(input_data)[0]

    st.divider()

    st.subheader("📈 Prediction Results")

    st.metric(
        label="Predicted Exam Score",
        value=f"{predicted_score:.2f} / 100"
    )

    st.write("")

    if predicted_status == 0:
        st.warning("⚠️ Needs Academic Support")

    else:
        st.success("✅ Satisfactory Performance")

    st.caption(
        "Note: The performance category uses the project's "
        "65-point threshold. It is not an official pass/fail result."
    )
