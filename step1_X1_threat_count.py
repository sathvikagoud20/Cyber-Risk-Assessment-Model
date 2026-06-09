import pandas as pd

# Load dataset
data = pd.read_csv("../data/cybersecurity_intrusion_data.csv")

# Attack column identified from dataset
attack_column = "attack_detected"

# Count attacks
X1 = data[data[attack_column] == 1].shape[0]

print("ΣX1 (Active Cyber Threat Vectors):", X1)
