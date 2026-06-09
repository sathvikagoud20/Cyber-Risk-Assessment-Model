import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
data = pd.read_csv("cybersecurity_intrusion_data.csv")

# Data Preprocessing: 
# Check missing values
print("Missing values per column:")
print(data.isnull().sum())

# Handle missing encryption values
data["encryption_used"] = data["encryption_used"].fillna("None")
print("After preprocessing:")
print(data.isnull().sum())


# Ensure numeric columns
numeric_cols = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "unusual_time_access",
    "attack_detected"
]

data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)