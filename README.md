# 💳 Credit Card Fraud Detection

> A machine learning project for **detecting fraudulent card transactions** in highly imbalanced financial data using **LightGBM**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-Gradient%20Boosted%20Trees-458B74?style=for-the-badge&logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-EDA-440154?style=for-the-badge&logo=seaborn&logoColor=white" />
</p>

---

## 📌 Problem Statement

Fraud detection is a **heavily imbalanced classification problem** where only **3.5%** of transactions are fraudulent.

**Goals:**

- 🎯 Accurately detect fraud  
- 🚨 Minimize false alarms (false positives)  
- 🧮 Optimize decision threshold based on **business tradeoffs**

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| 🐍 **Language** | Python |
| 📦 **Data & Prep** | Pandas, NumPy |
| 🤖 **Modeling** | LightGBM, Scikit-Learn |
| 📈 **Visualization** | Matplotlib, Seaborn |

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Transactions | 590,540 |
| Fraud Rate | 3.5% |
| Numeric Features | 332 |
| Categorical Features | 26 |
| **Total Features** | **358** |

Includes **behavioral (V features)**, transaction, device, card, and identity variables.

---

## 🔎 Exploratory Data Analysis (EDA)

- Confirmed **severe class imbalance**  
- Correlation analysis for numeric features vs target  
- Identified strong fraud indicators (**V258, V70, C14**)  
- Extracted feature importance trends for model interpretability  

---

## 🤖 Model Architecture

**Core Algorithm:** 🌲 **LightGBM (Gradient Boosted Trees)**

### ⚙️ Why LightGBM?

- Handles **missing values** and **large datasets** efficiently  
- Supports **categorical features natively**  
- Strong performance on **high-dimensional tabular data**  

### 📈 Model Performance

| Metric | Mean | Std |
|--------|------|-----|
| **ROC-AUC** | **0.9673** | ±0.0013 |
| **PR-AUC** | **0.8223** | ±0.0052 |

---

## 🎯 Threshold Optimization

Instead of using only the default **0.5** threshold, multiple decision thresholds were evaluated:

| Threshold | Precision | Recall    | F1-Score  | False Positives | Use Case                                |
|----------:|----------:|----------:|----------:|----------------:|-----------------------------------------|
| 0.05      | 0.084     | 0.978     | 0.156     | 43,713          | Maximum fraud capture (very aggressive) |
| 0.10      | 0.127     | 0.963     | 0.225     | 27,326          | Investigation-heavy systems             |
| 0.20      | 0.216     | 0.929     | 0.350     | 13,963          | High fraud recall focus                 |
| 0.30      | 0.311     | 0.897     | 0.462     | 8,208           | Risk-averse financial systems           |
| 0.40      | 0.414     | 0.861     | 0.559     | 5,046           | Balanced monitoring                     |
| 0.50      | **0.526** | **0.833** | **0.645** | 3,100           | Recommended balanced deployment         |

> ⚖️ This configuration **balances fraud detection with manageable false positives** for real-world use.

---

## 🧠 Key Insights

- 🧩 **Behavioral V-series features** are highly discriminative for fraud  
- 💻 Device and email-related features add strong predictive power  
- 💳 Transaction amount + card-level patterns are critical signals  
- 📉 Model shows **stable performance across folds**, indicating good generalization  

---

## 🏆 What This Project Demonstrates

- ✅ Handling **extreme class imbalance** in financial fraud data  
- ✅ Robust **train / validation** methodology with cross-validation  
- ✅ **Threshold tuning** for different business risk profiles  
- ✅ End-to-end ML workflow: **EDA → Modeling → Evaluation → Thresholding**

---


