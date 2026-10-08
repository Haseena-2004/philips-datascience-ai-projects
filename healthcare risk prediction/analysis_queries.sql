-- 1. Overall high-risk rate
SELECT COUNT(*) AS patients, ROUND(100.0*AVG(high_risk_outcome),1) AS high_risk_pct FROM patients;

-- 2. Risk by age band
SELECT CASE WHEN age<40 THEN '18-39' WHEN age<55 THEN '40-54' WHEN age<70 THEN '55-69' ELSE '70+' END AS age_band,
       COUNT(*) AS patients, ROUND(100.0*AVG(high_risk_outcome),1) AS high_risk_pct
FROM patients GROUP BY age_band ORDER BY MIN(age);

-- 3. Risk by smoking status and activity level
SELECT smoker, physical_activity, COUNT(*) AS patients, ROUND(100.0*AVG(high_risk_outcome),1) AS high_risk_pct
FROM patients GROUP BY smoker, physical_activity ORDER BY high_risk_pct DESC;

-- 4. Clinical indicator averages by outcome
SELECT high_risk_outcome, ROUND(AVG(systolic_bp),1) AS avg_bp, ROUND(AVG(cholesterol),1) AS avg_chol,
       ROUND(AVG(glucose),1) AS avg_glucose, ROUND(AVG(bmi),1) AS avg_bmi
FROM patients GROUP BY high_risk_outcome;

-- 5. Patients with 2+ chronic conditions AND 2+ prior admissions (care-management candidates)
SELECT COUNT(*) AS candidates, ROUND(100.0*AVG(high_risk_outcome),1) AS high_risk_pct
FROM patients WHERE chronic_conditions>=2 AND prior_admissions>=2;

-- 6. Data quality: missing values per column
SELECT SUM(bmi IS NULL) AS missing_bmi, SUM(cholesterol IS NULL) AS missing_chol, SUM(glucose IS NULL) AS missing_glucose FROM patients;
