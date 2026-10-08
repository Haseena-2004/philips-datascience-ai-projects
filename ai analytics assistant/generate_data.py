"""Generate a SIMULATED business-operations dataset (orders) and load it into SQLite."""
import sqlite3
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(11)
N = 12_000
HERE = Path(__file__).parent
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 640, N), unit="D")
region = rng.choice(["North", "South", "East", "West"], N, p=[.25, .3, .2, .25])
category = rng.choice(["Monitoring", "Imaging", "Diagnostics", "Home Care", "Accessories"], N, p=[.2, .15, .2, .25, .2])
channel = rng.choice(["Direct", "Distributor", "Online"], N, p=[.4, .35, .25])
units = rng.integers(1, 15, N)
unit_price = pd.Series(category).map({"Monitoring": 900, "Imaging": 2400, "Diagnostics": 700, "Home Care": 300, "Accessories": 60}).values
revenue = (units * unit_price * rng.uniform(.85, 1.1, N)).round(2)
cost = (revenue * rng.uniform(.55, .8, N)).round(2)
returned = rng.binomial(1, 0.04 + (category == "Accessories") * 0.03)
csat = np.clip(rng.normal(4.1 - returned * 1.2, .7, N), 1, 5).round(1)
delivery_days = np.clip(rng.normal(5 + (region == "West") * 1.5 + (channel == "Distributor") * 1, 2, N), 1, 20).round(0)
df = pd.DataFrame({"order_id": np.arange(1, N + 1), "order_date": dates.strftime("%Y-%m-%d"), "region": region,
                   "product_category": category, "channel": channel, "units": units, "revenue": revenue,
                   "cost": cost, "returned": returned, "csat_score": csat, "delivery_days": delivery_days})
(HERE / "data").mkdir(exist_ok=True)
with sqlite3.connect(HERE / "data" / "business.db") as con:
    df.to_sql("orders", con, if_exists="replace", index=False)
print(f"Saved {len(df):,} simulated orders to data/business.db")
