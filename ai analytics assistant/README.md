# AI-Powered Data Analytics Assistant

Ask business questions in plain English → get validated SQL, a result table, a chart, and a **grounded** natural-language insight.

**Stack:** Python · SQLite · LLMs / Generative AI (Claude API, optional) · Streamlit · pytest

## How it works
```
question ─▶ SQL generation (LLM, or rule-based fallback)
         ─▶ SQL validator (SELECT-only, single statement, table allow-list, forced LIMIT, no comments)
         ─▶ read-only SQLite execution
         ─▶ insight generation (LLM) ─▶ grounding check (every number must exist in the result table)
         ─▶ fallback to deterministic template if the check fails
```
**Design choices:** the LLM never gets direct database access; unsafe SQL is blocked *before* execution; the DB connection is read-only; LLM summaries that cite numbers not in the data are rejected. The app works with **no API key** (rule-based mode) so it is fully reproducible.

## Run
```bash
python generate_data.py          # 12,000 simulated orders -> data/business.db
pytest -q                        # 13 tests: SQL safety, grounding, end-to-end
streamlit run app.py
```
Optional LLM mode: `export ANTHROPIC_API_KEY=...` (model override via `ANTHROPIC_MODEL`).

## Example questions
*Total revenue by region · Return rate by product category · Monthly revenue trend · Average delivery days by channel in the West · Margin by category*

## Limitations
Rule-based fallback handles single-metric / single-dimension questions only; complex questions need LLM mode. Single table schema. Validation is regex-based — for production use a SQL parser (e.g. `sqlglot`) and a least-privilege DB user.
