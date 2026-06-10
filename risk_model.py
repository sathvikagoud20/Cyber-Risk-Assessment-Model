
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# ==========================================
# 1️⃣ LOAD DATASET
# ==========================================
file_path = "../data/cybersecurity_intrusion_data.csv"
# file_path = "UNSW_NB15_training-set.csv"

data = pd.read_csv(file_path)

print("\nDataset Loaded Successfully")
print("Shape:", data.shape)

if len(data) > 50000:
    data = data.sample(5000, random_state=42)
    print("Large dataset detected. Sampled 5000 rows for analysis.")

# (HANDLE LABEL SAFELY )
label_column = None

# Prefer numeric label column
if "label" in data.columns:
    label_column = "label"

elif "attack_detected" in data.columns:
    label_column = "attack_detected"

# Convert label to numeric
if label_column:
    if not pd.api.types.is_numeric_dtype(data[label_column]):
        data[label_column] = pd.factorize(data[label_column])[0]

print("Using label column:", label_column)

# ==========================================
# 3️⃣ NUMERIC COLUMN DETECTION
# ==========================================
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)

# ==========================================
# 4️⃣ AUTO DETECT THREAT & CONTROL FEATURES
# ==========================================
THREAT_KEYWORDS = ["attack", "login", "failed", "malicious",
                   "intrusion", "threat"]

CONTROL_KEYWORDS = ["reputation", "security", "patch",
                    "control", "defense", "protection"]

THREAT_COLUMNS = []
CONTROL_COLUMNS = []

for col in numeric_cols:
    col_lower = col.lower()

    if col == label_column:
        continue  
    if any(keyword in col_lower for keyword in THREAT_KEYWORDS):
        THREAT_COLUMNS.append(col)

    elif any(keyword in col_lower for keyword in CONTROL_KEYWORDS):
        CONTROL_COLUMNS.append(col)

# Fallback if nothing detected
if len(THREAT_COLUMNS) == 0:
    THREAT_COLUMNS = list(numeric_cols[:len(numeric_cols)//2])

if len(CONTROL_COLUMNS) == 0:
    CONTROL_COLUMNS = list(numeric_cols[len(numeric_cols)//2:])

print("\nDetected Threat Columns:", THREAT_COLUMNS)
print("Detected Control Columns:", CONTROL_COLUMNS)

# ==========================================
# 5️⃣ OUTLIER HANDLING
# ==========================================
for col in numeric_cols:
    data[col] = data[col].clip(
        lower=data[col].quantile(0.01),
        upper=data[col].quantile(0.99)
    )

# ==========================================
# 6️⃣ CALCULATE X1 & X2
# ==========================================
data["X1"] = data[THREAT_COLUMNS].sum(axis=1)
data["X2"] = data[CONTROL_COLUMNS].sum(axis=1)

# ==========================================
# 7️⃣ AUTO WEIGHT LEARNING
# ==========================================
if label_column:
    corr_x1 = abs(data["X1"].corr(data[label_column]))
    corr_x2 = abs(data["X2"].corr(data[label_column]))
else:
    corr_x1, corr_x2 = 0.6, 0.4

total_corr = corr_x1 + corr_x2

if total_corr == 0:
    C1, C2 = 0.6, 0.4
else:
    C1 = corr_x1 / total_corr
    C2 = corr_x2 / total_corr

C = 100

print("\n--- AUTO WEIGHT LEARNING ---")
print("Threat Weight (C1):", round(C1, 3))
print("Control Weight (C2):", round(C2, 3))

# ==========================================
# 8️⃣ CYBER RISK SCORE
# ==========================================
data["Cyber_Risk_Score"] = C - (C1 * data["X1"]) + (C2 * data["X2"])

# ==========================================
# 9️⃣ CIA MODEL
# ==========================================
data["Confidentiality"] = data["X1"]
data["Integrity"] = data["X1"]
data["Availability"] = data["X2"]

for col in ["Confidentiality", "Integrity", "Availability"]:
    if data[col].max() != 0:
        data[col] = (data[col] / data[col].max()) * 10

data["CIA_Score"] = data[["Confidentiality",
                          "Integrity",
                          "Availability"]].mean(axis=1)

data["Enhanced_Risk_Score"] = data["Cyber_Risk_Score"] - (data["CIA_Score"] * 0.5)

# ==========================================
# 🔟 RISK CLASSIFICATION
# ==========================================
def classify(score):
    if score >= 97:
        return "LOW RISK"
    elif score >= 94:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

data["Risk_Level"] = data["Enhanced_Risk_Score"].apply(classify)

# ==========================================
# 1️⃣1️⃣ OVERALL STATUS
# ==========================================
overall_score = data["Enhanced_Risk_Score"].mean()

if overall_score >= 97:
    overall_status = "LOW RISK"
    alert_message = "System stable."
elif overall_score >= 94:
    overall_status = "MEDIUM RISK"
    alert_message = "Moderate risk detected."
else:
    overall_status = "HIGH RISK"
    alert_message = "CRITICAL ALERT!"

print("\n--- OVERALL SYSTEM STATUS ---")
print("Average Cyber Risk Score:", round(overall_score,3))
print("Overall Risk Level:", overall_status)
print("Alert:", alert_message)

# ==========================================
# 1️⃣2️⃣ STATISTICAL VALIDATION
# ==========================================
mean = data["Enhanced_Risk_Score"].mean()
std = data["Enhanced_Risk_Score"].std()
n = len(data)

UCL = mean + 3*std
LCL = mean - 3*std

USL = 100
LSL = 90

Cp = (USL - LSL) / (6 * std)
Cpk = min((USL - mean)/(3*std), (mean - LSL)/(3*std))

# stat, p_value = stats.shapiro(data["Enhanced_Risk_Score"])
sample_data = data["Enhanced_Risk_Score"].sample(500)
stat, p_value = stats.shapiro(sample_data)

print("\n--- STATISTICAL REPORT ---")
print("Sample Size (n):", n)
print("Mean:", round(mean,3))
print("Std Dev:", round(std,3))
print("p-value:", round(p_value,6))
print("Cp:", round(Cp,3))
print("Cpk:", round(Cpk,3))

# ==========================================
# 1️⃣3️⃣ FULL GRAPH SECTION
# ==========================================

# 1️⃣ Gage R&R – Components of Variation
plt.figure()
data[numeric_cols].var().plot(kind="bar")
plt.title("Components of Variation (Gage R&R Approximation)")
plt.ylabel("Variance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2️⃣ Normal Probability Plot
plt.figure()
stats.probplot(data["Enhanced_Risk_Score"], dist="norm", plot=plt)
plt.title("Normal Probability Plot of Enhanced Risk Score")
plt.tight_layout()
plt.show()

# 3️⃣ X1 vs Enhanced Risk
plt.figure()
plt.scatter(data["X1"], data["Enhanced_Risk_Score"], alpha=0.6)

m1, b1 = np.polyfit(data["X1"], data["Enhanced_Risk_Score"], 1)
plt.plot(data["X1"], m1 * data["X1"] + b1)

plt.xlabel("Active Threat Vectors (X1)")
plt.ylabel("Enhanced Risk Score")
plt.title("X1 vs Enhanced Risk Score")
plt.tight_layout()
plt.show()

# 4️⃣ X2 vs Enhanced Risk
plt.figure()
plt.scatter(data["X2"], data["Enhanced_Risk_Score"], alpha=0.6)

m2, b2 = np.polyfit(data["X2"], data["Enhanced_Risk_Score"], 1)
plt.plot(data["X2"], m2 * data["X2"] + b2)

plt.xlabel("Security Controls (X2)")
plt.ylabel("Enhanced Risk Score")
plt.title("X2 vs Enhanced Risk Score")
plt.tight_layout()
plt.show()

# 5️⃣ Stability Chart
plt.figure()
plt.plot(data["Enhanced_Risk_Score"], marker='o', linestyle='')
plt.axhline(mean)
plt.axhline(UCL, linestyle="--")
plt.axhline(LCL, linestyle="--")
plt.title("Stability Chart")
plt.tight_layout()
plt.show()
...
