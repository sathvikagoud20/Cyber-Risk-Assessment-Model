import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# ===============================
# GLOBAL GRAPH SETTINGS ⭐
# ===============================
plt.style.use("ggplot")              # nice graph theme
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("../data/cybersecurity_intrusion_data.csv")

# ===============================
# 2. Data Preprocessing
# ===============================
print("Missing values per column:")
print(data.isnull().sum())

data["encryption_used"] = data["encryption_used"].fillna("None")

print("\nAfter preprocessing:")
print(data.isnull().sum())

# Convert to numeric
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

# Clip extreme values
data["session_duration"] = data["session_duration"].clip(
    lower=data["session_duration"].quantile(0.01),
    upper=data["session_duration"].quantile(0.99)
)

data["network_packet_size"] = data["network_packet_size"].clip(
    lower=data["network_packet_size"].quantile(0.01),
    upper=data["network_packet_size"].quantile(0.99)
)

# ===============================
# 3. Threat Vectors (X1)
# ===============================
data["X1"] = (
    data["login_attempts"]
    + data["failed_logins"]
    + data["unusual_time_access"]
    + data["attack_detected"]
)

# ===============================
# 4. Security Controls (X2)
# ===============================
data["encryption_flag"] = data["encryption_used"].apply(
    lambda x: 0 if x == "None" else 1
)

data["X2"] = data["encryption_flag"] + data["ip_reputation_score"]

# ===============================
# 5. Cyber Risk Score
# ===============================
C = 100
C1 = 0.6
C2 = 0.4

data["Cyber_Risk_Score"] = C - (C1 * data["X1"]) + (C2 * data["X2"])

# ===============================
# Risk Classification
# ===============================
def classify_risk(score):
    if score >= 97:
        return "LOW RISK"
    elif score >= 94:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

data["Risk_Level"] = data["Cyber_Risk_Score"].apply(classify_risk)

print("\n--- ALERT REPORT ---")
print("High Risk Sessions:", (data["Risk_Level"] == "HIGH RISK").sum())
print("Medium Risk Sessions:", (data["Risk_Level"] == "MEDIUM RISK").sum())
print("Low Risk Sessions:", (data["Risk_Level"] == "LOW RISK").sum())

# ===============================
# GRAPH 1: Components of Variation
# ===============================
data[[
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "failed_logins",
    "ip_reputation_score"
]].var().plot(kind="bar", color="skyblue")

plt.ylabel("Variance")
plt.title("Components of Variation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# GRAPH 2: Normal Probability Plot
# ===============================
plt.figure()
stats.probplot(data["Cyber_Risk_Score"], dist="norm", plot=plt)
plt.title("Normal Probability Plot of Cyber Risk Score")
plt.grid(True, color="lightgray")
plt.tight_layout()
plt.show()

stat, p_value = stats.shapiro(data["Cyber_Risk_Score"])
print("\nNormality Test p-value:", p_value)

# ===============================
# GRAPH 3: X1 vs Risk
# ===============================
plt.scatter(data["X1"], data["Cyber_Risk_Score"], alpha=0.6, color="orange")

m1, b1 = np.polyfit(data["X1"], data["Cyber_Risk_Score"], 1)
plt.plot(data["X1"], m1 * data["X1"] + b1, color="red", linewidth=2)

plt.xlabel("Active Threat Vectors (X1)")
plt.ylabel("Cyber Risk Score")
plt.title("X1 vs Cyber Risk Score")
plt.tight_layout()
plt.show()

# ===============================
# GRAPH 4: X2 vs Risk
# ===============================
plt.scatter(data["X2"], data["Cyber_Risk_Score"], alpha=0.6, color="green")

m2, b2 = np.polyfit(data["X2"], data["Cyber_Risk_Score"], 1)
plt.plot(data["X2"], m2 * data["X2"] + b2, color="blue", linewidth=2)

plt.xlabel("Security Controls (X2)")
plt.ylabel("Cyber Risk Score")
plt.title("X2 vs Cyber Risk Score")
plt.tight_layout()
plt.show()

# ===============================
# GRAPH 5: Stability Control Chart
# ===============================
mean = data["Cyber_Risk_Score"].mean()
std = data["Cyber_Risk_Score"].std()

UCL = mean + 3 * std
LCL = mean - 3 * std

plt.plot(data["Cyber_Risk_Score"], marker="o", linestyle="-", alpha=0.5, color="purple")

plt.axhline(mean, linestyle="-", linewidth=2, color="black", label="Mean")
plt.axhline(UCL, linestyle="--", linewidth=2, color="red", label="UCL (+3σ)")
plt.axhline(LCL, linestyle="--", linewidth=2, color="red", label="LCL (-3σ)")

plt.xlabel("Session Index")
plt.ylabel("Cyber Risk Score")
plt.title("Stability Control Chart")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# GRAPH 6: Capability Analysis
# ===============================
plt.hist(data["Cyber_Risk_Score"], bins=25, density=True, alpha=0.6, color="lightgreen")

x = np.linspace(mean - 4*std, mean + 4*std, 200)
plt.plot(x, stats.norm.pdf(x, mean, std), color="darkgreen", linewidth=2)

plt.xlabel("Cyber Risk Score")
plt.ylabel("Density")
plt.title("Process Capability Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# GRAPH 7: Risk Level Distribution
# ===============================
data["Risk_Level"].value_counts().plot(kind="bar", color=["red", "orange", "green"])

plt.xlabel("Risk Level")
plt.ylabel("Number of Sessions")
plt.title("Risk Level Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ===============================
# Save Final Report
# ===============================
data.to_csv("final_risk_report.csv", index=False)
print("\nFinal report saved as final_risk_report.csv")
