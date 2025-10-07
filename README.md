# 🇩🇪 German Credit Risk Prediction

### 🧠 Project Overview

This project aims to build an **interpretable credit risk prediction model** using the **German Credit Dataset (UCI Repository)**.
The goal is to help financial institutions assess the **probability of loan default** and improve decision-making accuracy through **explainable AI techniques**.

---

### 🎯 Business Objective

To develop a machine learning pipeline that:

* Predicts whether a loan applicant is likely to **default or repay**.
* Provides **transparent model insights** for risk analysts and credit officers.
* Demonstrates end-to-end **data preprocessing → modeling → interpretability** workflow.

---

### 🧰 Tech Stack

| Category         | Tools & Libraries               |
| ---------------- | ------------------------------- |
| Language         | Python                          |
| Data Handling    | pandas, numpy                   |
| Visualization    | matplotlib, seaborn             |
| Modeling         | scikit-learn, XGBoost           |
| Optimization     | Optuna, RandomizedSearchCV      |
| Interpretability | SHAP, Feature Importance        |
| Environment      | Google Colab, Jupyter Notebooks |

---

### 🗂️ Repository Structure

```
GermanCreditRiskPrediction/
│
├── data/                    # Raw and processed datasets
│   ├── german_credit_data.csv
│   ├── processed_credit_data.csv
│   └── scaled_credit_data.csv
│
├── models/                  # Trained models and scaler files
│   ├── xgb_optuna.pkl
│   ├── scaler.pkl
│   ├── randomsearchcv.pkl
│   ├── X_train.pkl, X_test.pkl
│   └── y_train.pkl, y_test.pkl
│
├── notebooks/               # Modularized notebooks
│   ├── A_DataPreprocessing.ipynb
│   ├── B_ExploratoryDataAnalysis.ipynb
│   ├── C_Modelling.ipynb
│   └── D_ModelDeployment&Monitoring.ipynb
│
├── README.md                # (You are here)
└── requirements.txt         # Dependencies for reproducibility
```

---

### 🚀 Project Workflow

| Step  | Notebook                             | Description                                                        |
| ----- | ------------------------------------ | ------------------------------------------------------------------ |
| **A** | `A_DataPreprocessing.ipynb`          | Data cleaning, encoding, scaling, and balancing                    |
| **B** | `B_ExploratoryDataAnalysis.ipynb`    | Statistical insights, visual exploration, feature importance       |
| **C** | `C_Modelling.ipynb`                  | Model training, hyperparameter tuning, comparison                  |
| **D** | `D_ModelDeployment&Monitoring.ipynb` | Model interpretability (SHAP), stability, and deployment readiness |

---

### 📊 Key Results

* **Best Model:** XGBoost (Optuna tuned)
* **ROC-AUC:** 0.88
* **Accuracy:** 82%
* **Top Predictors:** Checking account status, credit amount, loan duration, and savings account balance.

---

### 💡 Interpretability & Explainability

* **Global Interpretability:** Feature importance & SHAP summary plots
* **Local Interpretability:** SHAP force plots for individual applicants
* **Stability Checks:** Cross-validation and feature consistency validation

---

### 🧾 How to Reproduce

1. Clone this repository

   ```bash
   git clone https://github.com/<your-username>/GermanCreditRiskPrediction.git
   cd GermanCreditRiskPrediction
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run notebooks in order (A → D) to replicate results.

---

### 👤 Author

**Atharva Anirudha Karyakarte**
📧 [Mail]akaryakarte12@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/atharva-karyakarte)

---

### 🏦 Business Impact Summary

This project demonstrates how **AI-driven credit scoring** can enhance loan approval accuracy and transparency — crucial for regulatory compliance and fair lending practices.
It’s suitable for integration into a **bank’s risk management workflow** or as a **data science portfolio project** showcasing interpretability and responsible AI.

