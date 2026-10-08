# Customer Churn Prediction & Behavioral Analytics

**Goal:** go beyond descriptive analysis — connect **behaviour → ML prediction → recommended business action**.

**Stack:** Python · SQL (SQLite) · Pandas · Scikit-learn · Power BI

## Pipeline
1. `generate_data.py` – **8,000 simulated subscription customers** (tenure, contract, usage, support tickets, payments).
2. `sql/analysis_queries.sql` – churn by contract, tenure bucket, engagement, support friction; revenue lost by plan.
3. `run_pipeline.py` – engineered behavioural/engagement features (`engagement_score`, `friction_score`, `charge_per_login`, …); compares **Logistic Regression, Random Forest, Gradient Boosting** using 5-fold CV and a hold-out set (precision, recall, F1, ROC-AUC); churn-driver analysis via permutation importance; assigns risk tier × engagement tier and a **recommended retention action** per customer.

## Results (simulated data, seed 42)
| Model | CV ROC-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (selected) | 0.813 | 0.474 | 0.743 | 0.579 | 0.796 |
| Random Forest | 0.800 | 0.501 | 0.649 | 0.566 | 0.788 |
| Gradient Boosting | 0.800 | 0.476 | 0.709 | 0.569 | 0.791 |

Risk tiers on hold-out: actual churn ≈ 5% (Low), ≈ 18% (Medium), ≈ 47% (High). SQL highlights: month-to-month churn 37.9% vs 6.5% on two-year contracts; first-6-months churn 45.9%.

## Power BI dashboard
Import `outputs/powerbi_scored_customers.csv` and `outputs/powerbi_risk_segments.csv`. Suggested pages: churn overview, drivers (`churn_drivers.csv`), and a retention worklist filtered by `risk_tier` and `recommended_action`. *(Add `.pbix` + screenshots under `dashboard/`.)*

## Run
```bash
python generate_data.py && python run_pipeline.py
```

## Limitations
Simulated data; no temporal split or leakage audit as a production system would need. Retention-action rules are illustrative.
