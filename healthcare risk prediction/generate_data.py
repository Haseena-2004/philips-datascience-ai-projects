"""Generate a SIMULATED patient dataset (no real patient data) and load it into SQLite."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 10_000
HERE = Path(__file__).parent

age = rng.integers(18, 90, N)
gender = rng.choice(["Male", "Female"], N)
bmi = np.clip(rng.normal(27.5, 5.5, N), 15, 50).round(1)
systolic_bp = np.clip(rng.normal(125 + (age - 50) * 0.35, 16, N), 85, 210).round(0)
cholesterol = np.clip(rng.normal(195 + (age - 50) * 0.5, 38, N), 100, 350).round(0)
glucose = np.clip(rng.normal(105 + (bmi - 27) * 1.4, 26, N), 60, 300).round(0)
smoker = rng.choice([0, 1], N, p=[0.78, 0.22])
physical_activity = rng.choice(["Low", "Moderate", "High"], N, p=[0.38, 0.40, 0.22])
prior_admissions = rng.poisson(0.6 + (age > 65) * 0.8, N)
chronic_conditions = rng.poisson(0.5 + (age - 18) / 60, N).clip(0, 6)

act_effect = pd.Series(physical_activity).map({"Low": 0.5, "Moderate": 0.0, "High": -0.6}).values
logit = (-4.9 + 0.035 * age + 0.012 * (systolic_bp - 120) + 0.006 * (cholesterol - 190)
         + 0.010 * (glucose - 100) + 0.045 * (bmi - 25) + 0.65 * smoker + act_effect
         + 0.30 * prior_admissions + 0.35 * chronic_conditions)
p = 1 / (1 + np.exp(-logit))
high_risk = rng.binomial(1, p)

df = pd.DataFrame({
    "patient_id": np.arange(1, N + 1), "age": age, "gender": gender, "bmi": bmi,
    "systolic_bp": systolic_bp, "cholesterol": cholesterol, "glucose": glucose,
    "smoker": smoker, "physical_activity": physical_activity,
    "prior_admissions": prior_admissions, "chronic_conditions": chronic_conditions,
    "high_risk_outcome": high_risk,
})
# inject realistic missingness for the cleaning step
for col, frac in [("bmi", 0.03), ("cholesterol", 0.04), ("glucose", 0.02)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan

out = HERE / "data"
out.mkdir(exist_ok=True)
df.to_csv(out / "patients.csv", index=False)
with sqlite3.connect(out / "healthcare.db") as con:
    df.to_sql("patients", con, if_exists="replace", index=False)
print(f"Saved {len(df):,} simulated patient records. High-risk rate: {df.high_risk_outcome.mean():.1%}")
