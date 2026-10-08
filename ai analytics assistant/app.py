"""Streamlit UI:  streamlit run app.py"""
import sqlite3
import pandas as pd
import streamlit as st
from assistant.engine import DB_PATH, ask

st.set_page_config(page_title="AI Data Analytics Assistant", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Data Analytics Assistant")
st.caption("Ask business questions in plain English. SQL is generated, validated (read-only SELECT), executed, and explained.")

if not DB_PATH.exists():
    st.error("Database not found. Run `python generate_data.py` first."); st.stop()

with sqlite3.connect(DB_PATH) as con:
    kpi = pd.read_sql("SELECT COUNT(*) orders, SUM(revenue) revenue, SUM(revenue-cost) margin, 100.0*AVG(returned) ret, AVG(csat_score) csat FROM orders", con).iloc[0]
c = st.columns(5)
c[0].metric("Orders", f"{int(kpi.orders):,}"); c[1].metric("Revenue", f"${kpi.revenue/1e6:,.1f}M"); c[2].metric("Margin", f"${kpi.margin/1e6:,.1f}M")
c[3].metric("Return rate", f"{kpi.ret:.1f}%"); c[4].metric("Avg CSAT", f"{kpi.csat:.2f}")

examples = ["Total revenue by region", "Return rate by product category", "Average delivery days by channel",
            "Monthly revenue trend", "Margin by category", "Average CSAT by region"]
st.write("**Try:**")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex): st.session_state["q"] = ex
question = st.text_input("Your question", key="q", placeholder="e.g. Which region has the highest return rate?")

if question:
    with st.spinner("Thinking..."):
        r = ask(question)
    if not r["ok"]:
        st.error(r["error"]); st.code(r["sql"], language="sql")
    else:
        st.success(r["summary"]); st.caption(f"SQL via {r['engine']} · insight via {r['summary_source']}")
        left, right = st.columns([1, 1])
        with left: st.dataframe(r["data"], use_container_width=True)
        with right:
            d = r["data"]
            if d.shape[1] >= 2 and len(d) > 1: st.bar_chart(d.set_index(d.columns[0]).iloc[:, 0])
        with st.expander("Generated SQL"): st.code(r["sql"], language="sql")
