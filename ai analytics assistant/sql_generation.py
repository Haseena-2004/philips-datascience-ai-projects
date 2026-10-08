"""Natural language -> SQL. Uses an LLM when ANTHROPIC_API_KEY is set; otherwise a deterministic rule-based fallback
so the app (and the tests) work with zero setup."""
import os, re

SCHEMA = """Table orders(order_id INTEGER, order_date TEXT 'YYYY-MM-DD', region TEXT [North|South|East|West],
product_category TEXT [Monitoring|Imaging|Diagnostics|Home Care|Accessories], channel TEXT [Direct|Distributor|Online],
units INTEGER, revenue REAL, cost REAL, returned INTEGER (0/1), csat_score REAL (1-5), delivery_days REAL)
Margin = (revenue - cost). Return rate = AVG(returned)."""

SYSTEM_PROMPT = f"""You translate business questions into ONE SQLite SELECT query.
Schema:\n{SCHEMA}\nRules: output ONLY the SQL (no prose, no markdown); use only the orders table; read-only; aggregate where sensible;
alias columns clearly; for time trends use strftime('%Y-%m', order_date)."""

DIMENSIONS = {"region": "region", "category": "product_category", "product": "product_category", "channel": "channel",
              "month": "strftime('%Y-%m', order_date)", "quarter": "strftime('%Y', order_date) || '-Q' || ((cast(strftime('%m', order_date) as integer)+2)/3)"}
METRICS = {"revenue": "ROUND(SUM(revenue),0) AS total_revenue", "sales": "ROUND(SUM(revenue),0) AS total_revenue",
           "margin": "ROUND(SUM(revenue-cost),0) AS total_margin", "profit": "ROUND(SUM(revenue-cost),0) AS total_margin",
           "return": "ROUND(100.0*AVG(returned),2) AS return_rate_pct", "satisfaction": "ROUND(AVG(csat_score),2) AS avg_csat",
           "csat": "ROUND(AVG(csat_score),2) AS avg_csat", "delivery": "ROUND(AVG(delivery_days),2) AS avg_delivery_days",
           "units": "SUM(units) AS total_units", "orders": "COUNT(*) AS order_count"}


def rule_based_sql(question: str) -> str:
    q = question.lower()
    metric = next((v for k, v in METRICS.items() if k in q), METRICS["revenue"])
    dim = next((v for k, v in DIMENSIONS.items() if re.search(rf"\b{k}", q)), None)
    where = ""
    for col, vals in [("region", ["north", "south", "east", "west"]), ("channel", ["direct", "distributor", "online"])]:
        for v in vals:
            if re.search(rf"\b{v}\b", q):
                where = f" WHERE {col} = '{v.capitalize()}'"
    order = " ORDER BY 1" if dim and ("month" in q or "quarter" in q) else " ORDER BY 2 DESC"
    if dim:
        return f"SELECT {dim} AS {('period' if 'strftime' in dim else dim.split(' ')[0])}, {metric} FROM orders{where} GROUP BY 1{order}"
    return f"SELECT {metric} FROM orders{where}"


def generate_sql(question: str) -> tuple[str, str]:
    """Return (sql, engine) where engine is 'llm' or 'rule-based'."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=400,
                                         system=SYSTEM_PROMPT, messages=[{"role": "user", "content": question}])
            return msg.content[0].text.strip(), "llm"
        except Exception as e:  # network/key/model errors -> degrade gracefully
            print(f"[warn] LLM unavailable ({e}); using rule-based fallback")
    return rule_based_sql(question), "rule-based"
