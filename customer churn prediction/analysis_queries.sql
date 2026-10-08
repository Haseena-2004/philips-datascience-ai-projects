-- 1. Overall churn rate
SELECT COUNT(*) AS customers, ROUND(100.0*AVG(churned),1) AS churn_pct FROM customers;

-- 2. Churn by contract type
SELECT contract, COUNT(*) AS customers, ROUND(100.0*AVG(churned),1) AS churn_pct FROM customers GROUP BY contract ORDER BY churn_pct DESC;

-- 3. Churn by tenure bucket
SELECT CASE WHEN tenure_months<=6 THEN '0-6m' WHEN tenure_months<=12 THEN '7-12m' WHEN tenure_months<=24 THEN '13-24m' ELSE '25m+' END AS tenure_bucket,
       COUNT(*) AS customers, ROUND(100.0*AVG(churned),1) AS churn_pct
FROM customers GROUP BY tenure_bucket ORDER BY MIN(tenure_months);

-- 4. Engagement: low-login customers vs rest
SELECT CASE WHEN logins_30d<=8 THEN 'Low engagement (<=8 logins)' ELSE 'Healthy engagement' END AS engagement,
       COUNT(*) AS customers, ROUND(100.0*AVG(churned),1) AS churn_pct
FROM customers GROUP BY engagement;

-- 5. Support friction: tickets in last 90 days
SELECT support_tickets_90d, COUNT(*) AS customers, ROUND(100.0*AVG(churned),1) AS churn_pct
FROM customers WHERE support_tickets_90d<=5 GROUP BY support_tickets_90d ORDER BY support_tickets_90d;

-- 6. Revenue at risk (monthly charges of churned customers) by plan
SELECT plan, ROUND(SUM(CASE WHEN churned=1 THEN monthly_charges END),0) AS monthly_revenue_lost
FROM customers GROUP BY plan ORDER BY monthly_revenue_lost DESC;
