import streamlit as st
import pandas as pd
import plotly.express as px
import os
import random

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="Interactive Monitoring Dashboard",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD DATA FUNCTION
# -----------------------------------------------------------
@st.cache_data
def load_data(filepath):
    """
    Loads data from a CSV. If not found or empty, creates dummy data.
    """
    def create_dummy_data(path):
        if "patient_health" in path:
            dummy_data = {
                'timestamp': pd.date_range(start='2025-10-01 00:00', periods=24, freq='H'),
                'heart_rate': [random.randint(55, 130) for _ in range(24)],
                'oxygen_level': [random.randint(90, 100) for _ in range(24)]
            }
            df = pd.DataFrame(dummy_data)
        elif "student_scores" in path:
            subjects = ['Math', 'Science', 'English', 'History', 'Geography']
            data = {
                'timestamp': pd.date_range(start='2025-10-01', periods=30, freq='D'),
                'subject': [random.choice(subjects) for _ in range(30)],
                'score': [random.randint(20, 100) for _ in range(30)]
            }
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()
        df.to_csv(path, index=False)
        return df

    # Check for existing file
    if not os.path.exists(filepath):
        st.warning(f"⚠️ '{filepath}' not found. Creating dummy data.")
        return create_dummy_data(filepath)

    try:
        data = pd.read_csv(filepath)
        if data.empty:
            st.warning(f"'{filepath}' is empty. Creating dummy data.")
            return create_dummy_data(filepath)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
        return data
    except Exception as e:
        st.error(f"Error loading '{filepath}': {e}")
        return create_dummy_data(filepath)

# -----------------------------------------------------------
# LOAD DATASETS
# -----------------------------------------------------------
health_df = load_data("patient_health.csv")
scores_df = load_data("student_scores.csv")

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📂 Dashboard Options")
page = st.sidebar.radio("Select Dashboard", ["🏥 Health Monitoring", "📊 Student Scores"])

# -----------------------------------------------------------
# HEALTH MONITORING DASHBOARD
# -----------------------------------------------------------
if page == "🏥 Health Monitoring":
    st.title("🏥 Patient Health Monitoring Dashboard")

    # --- Heart Rate Chart ---
    st.subheader("💓 Heart Rate Over Time")
    fig_hr = px.line(
        health_df,
        x='timestamp',
        y='heart_rate',
        markers=True,
        title='Heart Rate Monitoring',
        labels={'timestamp': 'Time', 'heart_rate': 'Heart Rate (bpm)'},
        template="plotly_dark"
    )
    fig_hr.add_hline(y=120, line_dash="dash", line_color="red", annotation_text="High HR Threshold (120 bpm)")
    fig_hr.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Low HR Threshold (50 bpm)")
    st.plotly_chart(fig_hr, use_container_width=True)

    if (health_df['heart_rate'] > 120).any():
        st.warning("⚠️ Some heart rates exceeded the safe threshold (120 bpm).")
    if (health_df['heart_rate'] < 50).any():
        st.warning("⚠️ Some heart rates dropped below the safe threshold (50 bpm).")

    # --- Oxygen Level Chart ---
    st.subheader("🫁 Oxygen Level Over Time")
    fig_o2 = px.line(
        health_df,
        x='timestamp',
        y='oxygen_level',
        markers=True,
        title='Oxygen Level Monitoring',
        labels={'timestamp': 'Time', 'oxygen_level': 'Oxygen Saturation (%)'},
        template="plotly_dark"
    )
    fig_o2.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critical Level (95%)")
    st.plotly_chart(fig_o2, use_container_width=True)

    if (health_df['oxygen_level'] < 95).any():
        st.error("🚨 Critical Alert: Oxygen levels have dropped below 95%!")

# -----------------------------------------------------------
# STUDENT SCORES DASHBOARD
# -----------------------------------------------------------
elif page == "📊 Student Scores":
    st.title("📊 Student Scores Dashboard")

    # --- Score Trends Over Time ---
    st.subheader("📈 Score Trends Over Time")
    avg_scores = scores_df.groupby(scores_df['timestamp'].dt.date)["score"].mean().reset_index()
    fig_scores = px.line(
        avg_scores,
        x='timestamp',
        y='score',
        markers=True,
        title='Average Scores Over Time',
        labels={'timestamp': 'Date', 'score': 'Average Score'},
        template="plotly_dark"
    )
    fig_scores.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Fail Threshold (40%)")
    st.plotly_chart(fig_scores, use_container_width=True)

    if (avg_scores['score'] < 40).any():
        st.warning("⚠️ Some average scores fell below passing marks!")

    # --- Average Score by Subject ---
    st.subheader("📚 Average Scores by Subject")
    subject_avg = scores_df.groupby("subject")["score"].mean().reset_index()
    fig_subject = px.bar(
        subject_avg,
        x='subject',
        y='score',
        color='subject',
        title='Average Score by Subject',
        labels={'subject': 'Subject', 'score': 'Average Score'},
        template="plotly_dark"
    )
    st.plotly_chart(fig_subject, use_container_width=True)
