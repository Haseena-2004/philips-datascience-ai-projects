# Data Science & AI Engineering Portfolio

Three end-to-end projects covering the full data science workflow — **SQL analysis → feature engineering → machine learning → dashboards → Generative AI** — with a healthcare and business-analytics focus.

> **Data note:** all datasets are **synthetically generated** by the scripts in this repo (fixed random seeds, reproducible). No real patient or customer data is used. Model metrics therefore describe the simulated data, not real-world clinical or business performance.

| # | Project | Stack | Records |
|---|---------|-------|---------|
| 1 | [Healthcare Risk Prediction & Patient Analytics](healthcare-risk-prediction/) | Python, SQL, Pandas, Scikit-learn, Power BI | 10,000 |
| 2 | [Customer Churn Prediction & Behavioral Analytics](customer-churn-prediction/) | Python, SQL, Pandas, Scikit-learn, Power BI | 8,000 |
| 3 | [AI-Powered Data Analytics Assistant](ai-analytics-assistant/) | Python, SQL, LLMs (Claude API), Generative AI, Streamlit | 12,000 |

## Quick start
```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Project 1
cd healthcare-risk-prediction && python generate_data.py && python run_pipeline.py && cd ..
# Project 2
cd customer-churn-prediction && python generate_data.py && python run_pipeline.py && cd ..
# Project 3
cd ai-analytics-assistant && python generate_data.py && pytest -q && streamlit run app.py
```

## Skills demonstrated
Python · SQL (aggregations, CASE, CTEs) · Pandas/NumPy · Data cleaning & imputation · EDA · Feature engineering · Classification (Logistic Regression, Random Forest, Gradient Boosting) · Model evaluation (precision, recall, F1, ROC-AUC, cross-validation, permutation importance) · Power BI-ready data modelling · LLM integration, prompt design, output validation & guardrails · Streamlit · Testing (pytest) · Git/GitHub
