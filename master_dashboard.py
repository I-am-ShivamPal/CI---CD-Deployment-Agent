import streamlit as st
import pandas as pd
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="CI/CD Simulation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading ---
@st.cache_data(ttl=5)
def load_data():
    """Loads all data sources and returns them as dataframes."""
    data = {}
    files = {
        "deploy_log": "deployment_log.csv",
        "uptime": "uptime_timeline.csv",
        "scores": "student_scores.csv",
        "health": "patient_health.csv",
        "q_table": "q_table.csv"
    }
    for key, filename in files.items():
        if os.path.exists(filename):
            data[key] = pd.read_csv(filename)
        else:
            data[key] = pd.DataFrame()
    return data

# --- Load all data ---
data_frames = load_data()
deploy_log_df = data_frames["deploy_log"]
uptime_df = data_frames["uptime"]
scores_df = data_frames["scores"]
health_df = data_frames["health"]
q_table_df = data_frames["q_table"]

# --- Data Preprocessing ---
for df in [scores_df, health_df, deploy_log_df, uptime_df]:
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters ⚙️")
performance_view = st.sidebar.selectbox(
    "Select Performance View:",
    ["Student Scores", "Patient Health"]
)
remove_outliers = st.sidebar.checkbox("Remove Outliers")

# --- Main Dashboard ---
st.title("🤖 CI/CD Agent Simulation Master Dashboard")
if st.button("🔄 Refresh Data"):
    st.rerun()

tab1, tab2, tab3 = st.tabs(["📊 Performance & Event Timeline", "🧠 Agent Intelligence", " Raw Data Logs"])

# --- Tab 1: Performance & Event Timeline ---
with tab1:
    # --- DYNAMIC VIEW FOR STUDENT SCORES ---
    if performance_view == "Student Scores":
        st.header("Student Score Performance")
        if not scores_df.empty:
            df_to_plot = scores_df.sort_values('timestamp').copy()

            if remove_outliers and not df_to_plot.empty:
                Q1 = df_to_plot['score'].quantile(0.25)
                Q3 = df_to_plot['score'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                original_rows = len(df_to_plot)
                df_to_plot = df_to_plot[(df_to_plot['score'] >= lower_bound) & (df_to_plot['score'] <= upper_bound)]
                removed_count = original_rows - len(df_to_plot)
                st.info(f"Removed {removed_count} outlier(s).")

            if not df_to_plot.empty:
                df_to_plot['Success Event'] = np.nan
                df_to_plot['Failure Event'] = np.nan
                max_val = df_to_plot['score'].max()

                for _, row in deploy_log_df.iterrows():
                    if pd.notna(row['timestamp']):
                        time_diff = (df_to_plot['timestamp'] - row['timestamp']).abs()
                        if not time_diff.empty:
                            closest_idx = time_diff.idxmin()
                            if row['status'] == 'success':
                                df_to_plot.loc[closest_idx, 'Success Event'] = max_val
                            else:
                                df_to_plot.loc[closest_idx, 'Failure Event'] = max_val
                
                st.line_chart(df_to_plot.set_index('timestamp')[['score', 'Success Event', 'Failure Event']])
                st.info("🟢 Green spikes indicate successful deployments. 🔴 Red spikes indicate failed deployments.")
            else:
                st.warning("No data to display after outlier removal.")
        else:
            st.warning("`student_scores.csv` not found or is empty.")

    # --- DYNAMIC VIEW FOR PATIENT HEALTH ---
    elif performance_view == "Patient Health":
        st.header("Patient Health Performance")
        if not health_df.empty:
            col1, col2 = st.columns(2)
            
            health_to_plot = health_df.sort_values('timestamp').copy()

            if remove_outliers and not health_to_plot.empty:
                original_rows = len(health_to_plot)
                hr_q1, hr_q3 = health_to_plot['heart_rate'].quantile([0.25, 0.75])
                hr_iqr = hr_q3 - hr_q1
                health_to_plot = health_to_plot[(health_to_plot['heart_rate'] >= hr_q1 - 1.5 * hr_iqr) & (health_to_plot['heart_rate'] <= hr_q3 + 1.5 * hr_iqr)]
                
                o2_q1, o2_q3 = health_to_plot['oxygen_level'].quantile([0.25, 0.75])
                o2_iqr = o2_q3 - o2_q1
                health_to_plot = health_to_plot[(health_to_plot['oxygen_level'] >= o2_q1 - 1.5 * o2_iqr) & (health_to_plot['oxygen_level'] <= o2_q3 + 1.5 * o2_iqr)]
                removed_count = original_rows - len(health_to_plot)
                st.info(f"Removed {removed_count} health outlier(s).")

            if not health_to_plot.empty:
                health_to_plot['Success Event'] = np.nan
                health_to_plot['Failure Event'] = np.nan

                for _, row in deploy_log_df.iterrows():
                    if pd.notna(row['timestamp']):
                        time_diff = (health_to_plot['timestamp'] - row['timestamp']).abs()
                        if not time_diff.empty:
                            closest_idx = time_diff.idxmin()
                            if row['status'] == 'success':
                                health_to_plot.loc[closest_idx, 'Success Event'] = 1
                            else:
                                health_to_plot.loc[closest_idx, 'Failure Event'] = 1
                with col1:
                    st.subheader("Heart Rate")
                    hr_plot_df = health_to_plot.copy()
                    hr_plot_df['Success Event'] *= hr_plot_df['heart_rate'].max()
                    hr_plot_df['Failure Event'] *= hr_plot_df['heart_rate'].max()
                    st.line_chart(hr_plot_df.set_index('timestamp')[['heart_rate', 'Success Event', 'Failure Event']])

                with col2:
                    st.subheader("Oxygen Level")
                    o2_plot_df = health_to_plot.copy()
                    o2_plot_df['Success Event'] *= o2_plot_df['oxygen_level'].max()
                    o2_plot_df['Failure Event'] *= o2_plot_df['oxygen_level'].max()
                    st.line_chart(o2_plot_df.set_index('timestamp')[['oxygen_level', 'Success Event', 'Failure Event']])
                
                st.info("🟢 Green spikes indicate successful deployments. 🔴 Red spikes indicate failed deployments.")
            else:
                st.warning("No data to display after outlier removal.")
        else:
            st.warning("`patient_health.csv` not found or is empty.")

    # --- Combined Event Timeline ---
    st.header("Combined Event Timeline")
    if not deploy_log_df.empty:
        timeline_df = deploy_log_df.copy()
        if not uptime_df.empty:
             uptime_events = uptime_df.rename(columns={'event': 'details', 'status': 'action_type'})
             timeline_df = pd.concat([timeline_df, uptime_events], ignore_index=True)
        
        timeline_df.index.name = "Event ID"
        st.dataframe(timeline_df.sort_values(by='timestamp', ascending=False), use_container_width=True)
    else:
        st.warning("`deployment_log.csv` not found. Run the simulation.")

# --- Tab 2: Agent Intelligence ---
with tab2:
    st.header("Healing Agent Performance")
    
    if not deploy_log_df.empty:
        heal_actions = deploy_log_df[deploy_log_df['action_type'].str.contains('heal', na=False)]
        
        if not heal_actions.empty:
            success_rate = (heal_actions['status'] == 'success').sum() / len(heal_actions) * 100
            col1, col2 = st.columns(2)
            with col1: st.metric("Total Healing Attempts", len(heal_actions))
            with col2: st.metric("Healing Success Rate", f"{success_rate:.1f}%")
        else:
            st.info("No healing actions have been logged yet.")

    st.header("RL Planner Policy (Q-Table)")
    if not q_table_df.empty:
        st.info("This table shows the agent's learned score for each action. Higher scores are better.")
        
        def safe_formatter(x):
            try:
                return f"{float(x):.3f}"
            except (ValueError, TypeError):
                return x
        
        st.dataframe(q_table_df.style.background_gradient(cmap='viridis').format(safe_formatter))
    else:
        st.warning("`q_table.csv` not found. Run with `--planner rl` to generate it.")

# --- Tab 3: Raw Data Logs ---
with tab3:
    st.header("Raw Data Logs")
    with st.expander("Show Deployment Log"):
        st.dataframe(deploy_log_df, use_container_width=True)
    with st.expander("Show Uptime Timeline"):
        st.dataframe(uptime_df, use_container_width=True)
    with st.expander("Show Student Scores"):
        st.dataframe(scores_df, use_container_width=True)
    with st.expander("Show Patient Health Data"):
        st.dataframe(health_df, use_container_width=True)
    with st.expander("Show RL Q-Table"):
        st.dataframe(q_table_df, use_container_width=True)

