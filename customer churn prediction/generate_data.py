"""Generate a SIMULATED subscription-customer dataset (8,000 customers) and load it into SQLite."""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(7)
N = 8_000
HERE = Path(__file__).parent

tenure = rng.integers(1, 73, N)
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
plan = rng.choice(["Basic", "Standard", "Premium"], N, p=[0.35, 0.40, 0.25])
base = pd.Series(plan).map({"Basic": 25, "Standard": 45, "Premium": 75}).values
monthly_charges = (base + rng.normal(0, 6, N)).clip(15, 120).round(2)
logins_30d = np.clip(rng.poisson(14, N) - (contract == "Month-to-month") * 3, 0, None)
support_tickets_90d = rng.poisson(1.2, N)
late_payments_12m = rng.poisson(0.5, N)
payment_method = rng.choice(["Card", "Bank transfer", "Wallet", "Cash"], N, p=[0.45, 0.25, 0.2, 0.1])
discount_used = rng.choice([0, 1], N, p=[0.7, 0.3])
feature_adoption = np.clip(rng.beta(2.2, 2.5, N), 0, 1).round(2)

logit = (-0.6 - 0.035 * tenure + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
         - 0.07 * logins_30d + 0.28 * support_tickets_90d + 0.35 * late_payments_12m
         + 0.012 * (monthly_charges - 50) - 1.2 * feature_adoption + 0.4 * (payment_method == "Cash") - 0.25 * discount_used)
churn = rng.binomial(1, 1 / (1 + np.exp(-(logit + 0.9))))

df = pd.DataFrame({"customer_id": [f"C{100000+i}" for i in range(N)], "tenure_months": tenure, "contract": contract,
                   "plan": plan, "monthly_charges": monthly_charges, "logins_30d": logins_30d,
                   "support_tickets_90d": support_tickets_90d, "late_payments_12m": late_payments_12m,
                   "payment_method": payment_method, "discount_used": discount_used,
                   "feature_adoption": feature_adoption, "churned": churn})
df.loc[rng.choice(N, int(N * 0.02), replace=False), "monthly_charges"] = np.nan

(HERE / "data").mkdir(exist_ok=True)
df.to_csv(HERE / "data" / "customers.csv", index=False)
with sqlite3.connect(HERE / "data" / "churn.db") as con:
    df.to_sql("customers", con, if_exists="replace", index=False)
print(f"Saved {len(df):,} simulated customers. Churn rate: {df.churned.mean():.1%}")
