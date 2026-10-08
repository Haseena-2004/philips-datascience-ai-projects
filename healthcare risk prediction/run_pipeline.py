"""End-to-end pipeline: SQL analysis -> cleaning -> feature engineering -> model comparison -> Power BI exports."""
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

# ---- 1. SQL exploratory analysis ----
with sqlite3.connect(HERE / "data" / "healthcare.db") as con:
    queries = [q.strip() for q in (HERE / "sql" / "analysis_queries.sql").read_text().split(";") if "SELECT" in q]
    for i, q in enumerate(queries, 1):
        print(f"\n--- SQL query {i} ---\n{pd.read_sql(q, con).to_string(index=False)}")
    df = pd.read_sql("SELECT * FROM patients", con)

# ---- 2. Feature engineering ----
df["age_band"] = pd.cut(df.age, [0, 39, 54, 69, 120], labels=["18-39", "40-54", "55-69", "70+"])
df["pulse_pressure_proxy"] = df.systolic_bp - 80
df["metabolic_flag"] = ((df.bmi > 30).astype(int) + (df.glucose > 125).astype(int) + (df.cholesterol > 240).astype(int))
df["utilisation_score"] = df.prior_admissions + df.chronic_conditions
num = ["age", "bmi", "systolic_bp", "cholesterol", "glucose", "prior_admissions", "chronic_conditions",
       "metabolic_flag", "utilisation_score", "smoker"]
cat = ["gender", "physical_activity"]
X, y = df[num + cat], df["high_risk_outcome"]
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X, y, df.index, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(drop="first"), cat)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# ---- 3. Model comparison ----
results, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int) if name == "Gradient Boosting" else pipe.predict(X_te)
    results.append({"model": name, "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                    "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba)})
    fitted[name] = (pipe, proba, pred)
res = pd.DataFrame(results).round(3)
print("\n=== Model comparison (20% hold-out) ===\n", res.to_string(index=False))
res.to_csv(OUT / "model_comparison.csv", index=False)

best = res.sort_values("roc_auc", ascending=False).iloc[0]["model"]
pipe, proba, pred = fitted[best]
print(f"\nBest model by ROC-AUC: {best}\n", classification_report(y_te, pred, target_names=["Lower risk", "High risk"]))

# ---- 4. Charts ----
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for n, (_, p_, _) in fitted.items():
    fpr, tpr, _ = roc_curve(y_te, p_); ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, p_):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)

cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Lower", "High"], yticklabels=["Lower", "High"],
       xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {best}")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close(fig)

# feature importance via permutation (model-agnostic)
from sklearn.inspection import permutation_importance
pi = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=42)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 4.5)); imp.plot.barh(ax=ax, color="#0b5ed7")
ax.set(title="Permutation importance (drop in ROC-AUC)"); fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)
imp.sort_values(ascending=False).round(4).to_csv(OUT / "feature_importance.csv", header=["importance"])

# ---- 5. Power BI export ----
scored = df.loc[idx_te].copy()
scored["predicted_risk_score"] = proba.round(4)
scored["risk_segment"] = pd.cut(scored.predicted_risk_score, [-0.01, 0.2, 0.5, 1.01], labels=["Low", "Medium", "High"])
scored["age_band"] = scored["age_band"].astype(str); scored["risk_segment"] = scored["risk_segment"].astype(str)
scored.to_csv(OUT / "powerbi_scored_patients.csv", index=False)
kpi = pd.DataFrame({"kpi": ["Patients scored", "Actual high-risk rate", "Predicted High-segment share", "ROC-AUC", "Recall"],
                    "value": [len(scored), round(y_te.mean(), 3), round((scored.risk_segment == "High").mean(), 3),
                              round(float(res.set_index("model").loc[best, "roc_auc"]), 3),
                              round(float(res.set_index("model").loc[best, "recall"]), 3)]})
kpi.to_csv(OUT / "powerbi_kpis.csv", index=False)
json.dump({"best_model": best, **res.set_index("model").loc[best].to_dict(), "n_records": len(df)},
          open(OUT / "summary.json", "w"), indent=2)
print("\nSegment summary:\n", scored.groupby("risk_segment").agg(patients=("patient_id", "count"), actual_rate=("high_risk_outcome", "mean")).round(3))
print(f"\nSaved outputs to {OUT}")
