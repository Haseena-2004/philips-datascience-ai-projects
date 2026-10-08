"""SQL safety validation: the LLM is never trusted to run arbitrary SQL."""
import re

ALLOWED_TABLES = {"orders"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|truncate|grant|revoke)\b", re.I)


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str, max_rows: int = 200) -> str:
    """Return a safe, LIMIT-bounded SELECT, or raise UnsafeSQLError."""
    s = sql.strip().rstrip(";").strip()
    s = re.sub(r"^```(?:sql)?|```$", "", s, flags=re.I | re.M).strip()
    if ";" in s:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if "--" in s or "/*" in s:
        raise UnsafeSQLError("SQL comments are not allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(s):
        raise UnsafeSQLError("Query contains a forbidden keyword.")
    tables = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", s, re.I))
    ctes = set(re.findall(r"\b([a-zA-Z_]\w*)\s+as\s*\(", s, re.I))
    unknown = {t.lower() for t in tables} - ALLOWED_TABLES - {c.lower() for c in ctes}
    if unknown:
        raise UnsafeSQLError(f"Unknown table(s): {', '.join(sorted(unknown))}")
    if not re.search(r"\blimit\s+\d+", s, re.I):
        s += f" LIMIT {max_rows}"
    return s
