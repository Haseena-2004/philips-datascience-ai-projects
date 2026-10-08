import sqlite3
from pathlib import Path
import pandas as pd
from .insights import generate_insight
from .sql_generation import generate_sql
from .validator import UnsafeSQLError, validate_sql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "business.db"


def ask(question: str, db_path: Path = DB_PATH) -> dict:
    raw_sql, engine = generate_sql(question)
    try:
        sql = validate_sql(raw_sql)
    except UnsafeSQLError as e:
        return {"ok": False, "error": f"Blocked unsafe SQL: {e}", "sql": raw_sql, "engine": engine}
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:   # read-only connection
            df = pd.read_sql(sql, con)
    except Exception as e:
        return {"ok": False, "error": f"Query failed: {e}", "sql": sql, "engine": engine}
    summary, src = generate_insight(question, df)
    return {"ok": True, "sql": sql, "engine": engine, "data": df, "summary": summary, "summary_source": src}
