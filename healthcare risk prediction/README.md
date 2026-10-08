# Healthcare Risk Prediction & Patient Analytics

**Goal:** identify patients at higher predicted risk of an adverse outcome and turn model output into segments a care team can act on.

**Stack:** Python · SQL (SQLite) · Pandas · Scikit-learn · Power BI

## Pipeline
1. `generate_data.py` – creates **10,000 simulated patient records** (demographics, vitals, lifestyle, utilisation) with realistic missing values; loads them into SQLite.
2. `sql/analysis_queries.sql` – exploratory SQL: risk by age band, smoking × activity, clinical averages, care-management candidates, data-quality checks.
3. `run_pipeline.py` – median imputation, scaling, one-hot encoding; engineered features (`metabolic_flag`, `utilisation_score`); compares **Logistic Regression vs Gradient Boosting** on a stratified 80/20 split; reports precision / recall / F1 / ROC-AUC; permutation feature importance; writes Power BI files.

## Results (simulated data, seed 42)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.396 | 0.739 | 0.516 | 0.810 |
| Gradient Boosting | 0.567 | 0.281 | 0.376 | 0.806 |

Logistic Regression was selected: similar ROC-AUC, much higher recall — appropriate for a screening use-case where **missing a high-risk patient costs more than a false alarm**. Predicted risk tiers separate well on hold-out data: actual high-risk rate is ~3.6% (Low), ~11% (Medium) and ~40% (High).

Key SQL finding: risk rises from 6% (age 18–39) to ~40% (70+); smokers with low activity reach ~36%.

## Power BI dashboard
Import `outputs/powerbi_scored_patients.csv` and `outputs/powerbi_kpis.csv`. Suggested pages: (1) KPI cards + risk-segment donut, (2) risk by age band / activity / smoker, (3) patient drill-through table sorted by `predicted_risk_score`. *(Add your `.pbix` and screenshots to a `dashboard/` folder.)*

## Run
```bash
python generate_data.py && python run_pipeline.py
```
Outputs (`outputs/`): `model_comparison.csv`, `roc_curve.png`, `confusion_matrix.png`, `feature_importance.png/.csv`, `powerbi_*.csv`, `summary.json`.

## Limitations
Synthetic data with known generating relationships; not clinically validated and not for medical use. A real deployment would need fairness checks, calibration, and clinical review.
