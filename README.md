# Cyber Risk Assessment Model for Critical Information Infrastructure

## Overview

This project presents a Cyber Risk Assessment Model designed for Critical Information Infrastructure (CII) such as banking, healthcare, transportation, communication, and power systems. The model identifies cyber threats, analyzes vulnerabilities, computes risk scores, and classifies risks into Low, Medium, and High categories.

## Objectives

* Identify and analyze cyber threats and vulnerabilities.
* Develop a data-driven cyber risk evaluation framework.
* Calculate risk scores using threat and control factors.
* Apply CIA (Confidentiality, Integrity, Availability) impact analysis.
* Classify cyber risks into Low, Medium, and High levels.
* Generate statistical reports and visualizations for decision-making.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* VS Code

## Dataset

The project uses a cybersecurity intrusion detection dataset containing network activity, login attempts, failed logins, IP reputation scores, and attack indicators.

## Methodology

1. Data Preprocessing

   * Handle missing values
   * Remove duplicates
   * Normalize data

2. Feature Engineering

   * Detect threat-related features (X1)
   * Detect security control features (X2)

3. Intelligent Weight Learning

   * Automatically calculate weights based on feature correlation

4. Risk Computation

   * Calculate Cyber Risk Score
   * Apply CIA-based enhancement

5. Risk Classification

   * LOW RISK
   * MEDIUM RISK
   * HIGH RISK

6. Statistical Validation

   * Mean and Standard Deviation
   * Shapiro Normality Test
   * Control Limits (UCL/LCL)
   * Process Capability Analysis (Cp, Cpk)

## Results

* Dataset Size: 9,537 records
* Average Cyber Risk Score: 94.3
* Overall Risk Level: MEDIUM RISK
* Threat Weight (C1): 0.665
* Control Weight (C2): 0.335

## Visualizations

* Components of Variation
* Normal Probability Plot
* X1 vs Enhanced Risk Score
* X2 vs Enhanced Risk Score
* Stability Chart
* Capability Analysis
* Risk Level Distribution
* CIA Impact Score Distribution

## Future Enhancements

* Real-time packet analysis using Scapy/Zeek
* SCADA/ICS integration
* Machine Learning-based risk prediction
* Flask or Power BI dashboard
* Multi-dataset validation
* SOAR integration for automated response


## Project Outputs

### Components of Variation
![Components of Variation](components-of-variation.png)

### Normal Probability Plot
![Normal Probability Plot](normal-probability-plot.png)

### X1 vs Enhanced Risk Score
![X1 vs Enhanced Risk Score](x1-vs-enhanced-risk-score.png)

### X2 vs Enhanced Risk Score
![X2 vs Enhanced Risk Score](x2-vs-enhanced-risk-score.png)

### Stability Chart
![Stability Chart](stability-chart.png)

## Author

Chinthakula Sathvika
B.Tech CSE (Cyber Security)
CVR College of Engineering
Aspiring SOC Analyst
