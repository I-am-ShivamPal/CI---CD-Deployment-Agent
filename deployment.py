import pandas as pd
import random
import time
import datetime
import subprocess
import os
import csv
import argparse
import shutil
import numpy as np

# --- Agent 1: Deployment Logger ---
class DeployAgent:
    """Tracks and logs deployment events to a CSV file."""
    def __init__(self, log_file="deployment_log.csv"):
        self.log_file = log_file
        self._initialize_log_file()

    def _initialize_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "dataset_changed", "status", "response_time_ms", "action_type"])
            print(f"Initialized log file: {self.log_file}")

    def log_deployment(self, dataset, status, response_time, action_type="deploy"):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, dataset, status, round(response_time, 2), action_type])
        print(f"Logged {action_type} for {dataset}: {status} ({round(response_time,2)} ms)")


# --- Agent 2: Issue Detector ---
class IssueDetector:
    """Reads logs and data to determine the specific type of failure."""
    def __init__(self, log_file="deployment_log.csv", data_file="student_scores.csv"):
        self.log_file = log_file
        self.data_file = data_file
        self.latency_threshold_ms = 16000
        # Domain-specific thresholds
        self.low_score_threshold = 40
        self.high_hr_threshold = 120
        self.low_o2_threshold = 95

    def detect_failure_type(self):
        """
        Returns (state, message).
        Checks for deployment failures first, then data-specific anomalies.
        """
        try:
            # 1. Check for deployment failures from the log file
            df = pd.read_csv(self.log_file)
            df.dropna(how='all', inplace=True)
            if df.empty:
                # No deployment logs yet, so check the data file for anomalies
                pass
            else:
                df['response_time_ms'] = pd.to_numeric(df['response_time_ms'], errors='coerce')
                last_event = df.iloc[-1]
                status = str(last_event.get('status', '')).lower().strip()
                rt = last_event.get('response_time_ms')

                if status == 'failure':
                    return "deployment_failure", "Last deployment attempt failed."
                if pd.notna(rt) and rt > self.latency_threshold_ms:
                    return "latency_issue", f"High latency detected: {rt:.2f} ms."

            # 2. If no deployment issues, check for data anomalies in the relevant file
            if os.path.exists(self.data_file):
                if "student_scores" in self.data_file:
                    score_df = pd.read_csv(self.data_file)
                    if 'score' in score_df.columns and not score_df.empty:
                        avg_score = score_df['score'].mean()
                        if avg_score < self.low_score_threshold:
                            return "anomaly_score", f"Low student performance detected. Avg score={avg_score:.2f}"
                elif "patient_health" in self.data_file:
                    health_df = pd.read_csv(self.data_file)
                    if not health_df.empty:
                        last_vitals = health_df.iloc[-1]
                        if last_vitals.get('heart_rate', 0) > self.high_hr_threshold:
                            return "anomaly_health", f"Abnormal vital sign: High heart rate detected ({last_vitals['heart_rate']})."
                        if last_vitals.get('oxygen_level', 100) < self.low_o2_threshold:
                            return "anomaly_health", f"Abnormal vital sign: Low oxygen detected ({last_vitals['oxygen_level']})."

            return "no_failure", "No issues detected."
        except FileNotFoundError:
            return "no_failure", "Log file not found."
        except Exception as e:
            return "no_failure", f"Error analyzing logs: {e}"


# --- Agent 3: Uptime Monitor ---
class UptimeMonitor:
    """Maintains a synthetic uptime/downtime timeline."""
    def __init__(self, timeline_file="uptime_timeline.csv"):
        self.timeline_file = timeline_file
        self.last_status = self._get_initial_status()
        if self.last_status is None:
            print(f"Initialized uptime timeline: {self.timeline_file}")
            self.update_status("UP", "Initial status check")

    def _get_initial_status(self):
        if not os.path.exists(self.timeline_file):
            with open(self.timeline_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'status', 'event'])
            return None
        else:
            with open(self.timeline_file, 'r', newline='') as f:
                rows = list(csv.reader(f))
                return rows[-1][1] if len(rows) > 1 else None

    def update_status(self, new_status, event_description):
        if new_status != self.last_status:
            timestamp = datetime.datetime.now().isoformat()
            with open(self.timeline_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, new_status, event_description])
            print(f"Uptime Monitor: Service status changed to {new_status}. Reason: {event_description}")
            self.last_status = new_status
        else:
            print(f"Uptime Monitor: Service status remains {self.last_status}.")


# --- Agent 4 & 5: Healing Planners ---
class HealingPlanner:
    """A simple planner that randomly chooses a healing strategy."""
    # This class remains for comparison and as a fallback
    pass

class RLPlanner:
    """Uses Q-learning to choose the best healing strategy for a given failure type."""
    def __init__(self, q_table_file="q_table.csv"):
        self.q_table_file = q_table_file
        # Expanded states to handle data-specific anomalies
        self.states = ["deployment_failure", "latency_issue", "anomaly_score", "anomaly_health"]
        self.actions = ["retry_deployment", "restore_previous_version", "adjust_thresholds"]
        self.alpha, self.gamma, self.epsilon = 0.1, 0.9, 0.1
        self.q_table = self._load_q_table()
        print("Initialized RL Planner Agent.")

    def _load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                qt = pd.read_csv(self.q_table_file, index_col=0)
            except Exception:
                qt = pd.DataFrame()
        else:
            qt = pd.DataFrame()

        for s in self.states:
            if s not in qt.index: qt.loc[s] = 0.0
        for a in self.actions:
            if a not in qt.columns: qt[a] = 0.0
        
        return qt.loc[self.states, self.actions].fillna(0.0).astype(float)

    def save_q_table(self):
        print(f"\nSaving updated Q-table to {self.q_table_file}")
        self.q_table.to_csv(self.q_table_file)
        print("Save complete. The Q-Table is now:")
        print(self.q_table)

    def choose_action(self, state):
        if state not in self.q_table.index or random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
            print(f"RL Planner: Exploring -> Randomly chose action '{action}'")
        else:
            action = self.q_table.loc[state].idxmax()
            print(f"RL Planner: Exploiting -> Chose best action '{action}'")
        return action

    def update_q_table(self, state, action, reward):
        if state not in self.q_table.index:
            self.q_table.loc[state] = 0.0
        old_value = float(self.q_table.loc[state, action])
        new_value = old_value + self.alpha * (reward - old_value)
        self.q_table.loc[state, action] = new_value
        print(f"RL Planner: Q-table updated for state '{state}', action '{action}'. {old_value:.3f} -> {new_value:.3f}")

    def attempt_healing(self, state, dataset_path):
        action = self.choose_action(state)
        print(f"\n--- RL Planner: Initiating recovery for state '{state}' ---")
        print(f"Chosen Strategy: {action}")
        if action == 'retry_deployment':
            status, response_time = self._retry_deployment()
            return status, response_time, "heal_retry", action
        elif action == 'restore_previous_version':
            status, response_time = self._restore_previous_version(dataset_path)
            return status, response_time, "heal_restore", action
        elif action == 'adjust_thresholds':
            status, response_time = self._adjust_thresholds()
            return status, response_time, "heal_adjust", action
        return "failure", 0, "unknown_strategy", action

    def _retry_deployment(self): return trigger_dashboard_deployment(should_fail=False)
    def _restore_previous_version(self, dataset_path):
        if os.path.exists(f"{dataset_path}.bak"):
            shutil.copyfile(f"{dataset_path}.bak", dataset_path)
            return trigger_dashboard_deployment(should_fail=False)
        return "failure", 0
    def _adjust_thresholds(self): return "success", 200

# --- Simulation Functions ---
def simulate_data_change(dataset_path, force_anomaly=False):
    """Creates a backup and appends a new row, optionally forcing an anomaly."""
    print(f"\nSimulating change for '{dataset_path}'...")
    try:
        if not os.path.exists(dataset_path):
            # Create a dummy file if one doesn't exist
            print(f"Dataset '{dataset_path}' not found. Creating a dummy file.")
            if "student_scores" in dataset_path:
                pd.DataFrame([{'timestamp': '2025-01-01', 'name': 'Initial', 'subject': 'Math', 'score': 80}]).to_csv(dataset_path, index=False)
            elif "patient_health" in dataset_path:
                pd.DataFrame([{'timestamp': '2025-01-01', 'heart_rate': 75, 'blood_pressure': '120/80', 'oxygen_level': 98}]).to_csv(dataset_path, index=False)
        
        shutil.copyfile(dataset_path, f"{dataset_path}.bak")
        print(f"  -> Created backup: {dataset_path}.bak")

        df = pd.read_csv(dataset_path)

        if "student_scores" in dataset_path:
            new_row = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'name': random.choice(['Alice', 'Bob', 'Charlie', 'David']),
                'subject': random.choice(['Math', 'Science', 'History', 'English']),
                'score': 10 if force_anomaly else random.randint(50, 100)
            }
            print("  -> Added new student score record." + (" (ANOMALY FORCED)" if force_anomaly else ""))
        elif "patient_health" in dataset_path:
            new_row = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': 150 if force_anomaly else random.randint(60, 100),
                'blood_pressure': f"{random.randint(110, 140)}/{random.randint(70, 90)}",
                'oxygen_level': 90 if force_anomaly else random.randint(96, 100)
            }
            print("  -> Added new patient health record." + (" (ANOMALY FORCED)" if force_anomaly else ""))
        else:
            return

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(dataset_path, index=False)
        print(f"  -> Saved new data to '{dataset_path}'.")

    except Exception as e:
        print(f"Error simulating data change: {e}")

def trigger_dashboard_deployment(timeout=15, should_fail=False, failure_type='crash'):
    print("Triggering dashboard deployment...")
    if should_fail:
        if failure_type == 'crash':
            print("  -> SIMULATING DEPLOYMENT FAILURE (CRASH).")
            return "failure", 2000
        elif failure_type == 'latency':
            print("  -> SIMULATING DEPLOYMENT SUCCESS (HIGH LATENCY).")
            return "success", (timeout + 5) * 1000

    status, process = "failure", None
    start_time = time.time()
    try:
        command = ["streamlit", "run", "patient_health_and_student_scores.py", "--server.runOnSave", "false"]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(timeout)
        if process.poll() is None: status = "success"
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except Exception: pass
    return status, (time.time() - start_time) * 1000

# --- Main Simulation Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI/CD Simulation with Selectable Healing Agents")
    parser.add_argument("--dataset", type=str, default="student_scores.csv", help="Dataset to simulate (student_scores.csv or patient_health.csv)")
    parser.add_argument("--fail-type", type=str, choices=['crash', 'latency'], help="Force a deployment failure.")
    parser.add_argument("--force-anomaly", action="store_true", help="Force a data anomaly in the dataset.")
    parser.add_argument("--planner", type=str, choices=['random', 'rl'], default='rl', help="Choose the healing planner to use.")
    args = parser.parse_args()

    # Initialize Agents
    deploy_agent = DeployAgent()
    issue_detector = IssueDetector(data_file=args.dataset)
    uptime_monitor = UptimeMonitor()
    planner = RLPlanner() # Defaulting to RL Planner as it's the most advanced

    simulate_data_change(args.dataset, force_anomaly=args.force_anomaly)
    
    # Initial Deployment
    should_fail_flag = args.fail_type is not None
    deployment_status, deployment_time = trigger_dashboard_deployment(should_fail=should_fail_flag, failure_type=args.fail_type)
    deploy_agent.log_deployment(args.dataset, deployment_status, deployment_time)

    # Detect & Heal
    failure_state, reason = issue_detector.detect_failure_type()
    if failure_state != "no_failure":
        print(f"\nIssue Detected: {reason}")
        uptime_monitor.update_status("DOWN", reason)
        
        heal_status, heal_time, heal_type, chosen_action = planner.attempt_healing(failure_state, args.dataset)
        reward = 1 if heal_status == 'success' else -1
        planner.update_q_table(failure_state, chosen_action, reward)
        
        deploy_agent.log_deployment(args.dataset, heal_status, heal_time, action_type=heal_type)
        
        if heal_status == 'success':
            uptime_monitor.update_status("UP", "Recovery successful via " + heal_type)
        else:
            print("\n--- Healing attempt failed. The service remains down. ---")
    else:
        uptime_monitor.update_status("UP", "Successful deployment")

    planner.save_q_table()
    
    print("\nCI/CD simulation finished.")