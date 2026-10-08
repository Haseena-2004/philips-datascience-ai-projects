"""Churn pipeline: SQL EDA -> feature engineering -> model comparison (CV + hold-out) -> Power BI exports."""
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

with sqlite3.connect(HERE / "data" / "churn.db") as con:
    for i, q in enumerate([q.strip() for q in (HERE / "sql" / "analysis_queries.sql").read_text().split(";") if "SELECT" in q], 1):
        print(f"\n--- SQL query {i} ---\n{pd.read_sql(q, con).to_string(index=False)}")
    df = pd.read_sql("SELECT * FROM customers", con)

# ---- Feature engineering (behavioural + engagement) ----
df["tickets_per_month_tenure"] = df.support_tickets_90d / (df.tenure_months.clip(lower=1) / 3).clip(lower=1)
df["engagement_score"] = (df.logins_30d / 14) * 0.6 + df.feature_adoption * 0.4
df["friction_score"] = df.support_tickets_90d + 2 * df.late_payments_12m
df["is_new_customer"] = (df.tenure_months <= 6).astype(int)
df["charge_per_login"] = df.monthly_charges / (df.logins_30d + 1)

num = ["tenure_months", "monthly_charges", "logins_30d", "support_tickets_90d", "late_payments_12m", "discount_used",
       "feature_adoption", "tickets_per_month_tenure", "engagement_score", "friction_score", "is_new_customer", "charge_per_login"]
cat = ["contract", "plan", "payment_method"]
X, y = df[num + cat], df.churned
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X, y, df.index, test_size=0.2, stratify=y, random_state=42)

def make(clf, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    pre = ColumnTransformer([("num", Pipeline(steps), num), ("cat", OneHotEncoder(drop="first"), cat)])
    return Pipeline([("pre", pre), ("clf", clf)])

models = {
    "Logistic Regression": make(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": make(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1), scale=False),
    "Gradient Boosting": make(GradientBoostingClassifier(random_state=42), scale=False),
}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
rows, probas = [], {}
for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr); p = pipe.predict_proba(X_te)[:, 1]
    thr = 0.5 if name != "Gradient Boosting" else y_tr.mean()      # GB is not class-balanced -> use prevalence threshold
    pred = (p >= thr).astype(int); probas[name] = p
    rows.append({"model": name, "cv_roc_auc": cv_auc, "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                 "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, p)})
res = pd.DataFrame(rows).round(3)
print("\n=== Model comparison ===\n", res.to_string(index=False))
res.to_csv(OUT / "model_comparison.csv", index=False)
best = res.sort_values("roc_auc", ascending=False).iloc[0]["model"]; pipe = models[best]; p = probas[best]
print("Best model:", best)

# ---- Charts ----
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for n, pr in probas.items():
    fpr, tpr, _ = roc_curve(y_te, pr); ax.plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_te, pr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlabel="FPR", ylabel="TPR", title="Churn model ROC curves"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)

pi = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=42)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 5)); imp.plot.barh(ax=ax, color="#0b5ed7"); ax.set(title="Top churn drivers (permutation importance)")
fig.tight_layout(); fig.savefig(OUT / "churn_drivers.png", dpi=150); plt.close(fig)
imp.sort_values(ascending=False).round(4).to_csv(OUT / "churn_drivers.csv", header=["importance"])

# ---- Segmentation + Power BI exports ----
scored = df.loc[idx_te].copy()
scored["churn_probability"] = p.round(4)
scored["risk_tier"] = pd.cut(scored.churn_probability, [-.01, .25, .5, 1.01], labels=["Low", "Medium", "High"]).astype(str)
scored["engagement_tier"] = pd.qcut(scored.engagement_score, 3, labels=["Low", "Medium", "High"]).astype(str)
def action(r):
    if r.risk_tier == "High" and r.engagement_tier == "Low": return "Win-back outreach + onboarding re-engagement"
    if r.risk_tier == "High": return "Retention offer / contract upgrade"
    if r.risk_tier == "Medium" and r.support_tickets_90d >= 3: return "Proactive support follow-up"
    if r.risk_tier == "Medium": return "Feature-adoption nudge"
    return "Standard lifecycle communication"
scored["recommended_action"] = scored.apply(action, axis=1)
scored.to_csv(OUT / "powerbi_scored_customers.csv", index=False)
seg = scored.groupby("risk_tier").agg(customers=("customer_id", "count"), actual_churn=("churned", "mean"),
                                       monthly_revenue=("monthly_charges", "sum")).round(3)
seg.to_csv(OUT / "powerbi_risk_segments.csv"); print("\nRisk tiers:\n", seg)
json.dump({"best_model": best, **res.set_index("model").loc[best].to_dict(), "n_records": len(df)}, open(OUT / "summary.json", "w"), indent=2)
print("\nSaved outputs to", OUT)
