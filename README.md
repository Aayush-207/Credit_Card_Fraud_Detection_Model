# 💳 Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent transactions in highly imbalanced financial data using **LightGBM**.

## 📌 Problem Statement

Fraud detection is a classic imbalanced classification problem where only **3.5%** of transactions are fraudulent.

**Goals:**
- Accurately detect fraud
- Minimize false alarms 
- Optimize decision threshold based on business tradeoffs

## 📌 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-458B74?style=flat&logo=lightgbm&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-440154?style=flat&logo=seaborn&logoColor=white)

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Transactions | 590,540 |
| Fraud Rate | 3.5% |
| Numeric Features | 332 |
| Categorical Features | 26 |
| **Total Features** | **358** |

Includes behavioral (V features), transaction, device, card, and identity variables.

## 🔎 Exploratory Data Analysis (EDA)

- Verified severe class imbalance
- Analyzed numeric feature correlation with target
- Identified strong fraud indicators (**V258, V70, C14**)
- Examined feature importance trends

## 🤖 Model Architecture

**Algorithm:** LightGBM (Gradient Boosted Trees)

### Why LightGBM?
- Handles missing values natively
- Efficient with large datasets
- Supports categorical features natively
- Strong performance on tabular data


### Model Performance
| Metric | Mean | Std |
|--------|------|-----|
| **ROC-AUC** | **0.9673** | ±0.0013 |
| **PR-AUC** | **0.8223** | ±0.0052 |

## 🎯 Threshold Optimization

Instead of relying only on the default 0.5 threshold, multiple decision thresholds were evaluated.

| Threshold | Precision | Recall    | F1-Score  | False Positives | Use Case                                |
| --------- | --------- | --------- | --------- | --------------- | --------------------------------------- |
| 0.05      | 0.084     | 0.978     | 0.156     | 43,713          | Maximum fraud capture (very aggressive) |
| 0.10      | 0.127     | 0.963     | 0.225     | 27,326          | Investigation-heavy systems             |
| 0.20      | 0.216     | 0.929     | 0.350     | 13,963          | High fraud recall focus                 |
| 0.30      | 0.311     | 0.897     | 0.462     | 8,208           | Risk-averse financial systems           |
| 0.40      | 0.414     | 0.861     | 0.559     | 5,046           | Balanced monitoring                     |
| 0.50      | **0.526** | **0.833** | **0.645** | 3,100           | Recommended balanced deployment         |


*Balances fraud detection with manageable false positives*

## 🧠 Key Insights

- **Behavioral features (V-series)** strongly drive fraud detection
- Device and email-related features are highly predictive
- Transaction amount + card-level patterns contribute significantly
- Model remains stable across folds (strong generalization)

## 🏆 What This Project Demonstrates

- ✅ Handling extreme imbalanced classification
- ✅ Proper train/validation methodology
- ✅ Cross-validation for stability
- ✅ Threshold optimization for business use
- ✅ Complete ML workflow (EDA → Evaluation)



