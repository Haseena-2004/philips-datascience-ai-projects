import pandas as pd, pytest
from assistant.engine import ask
from assistant.insights import grounded
from assistant.validator import UnsafeSQLError, validate_sql


@pytest.mark.parametrize("bad", ["DROP TABLE orders", "SELECT * FROM orders; DELETE FROM orders", "UPDATE orders SET revenue=0",
                                 "SELECT * FROM sqlite_master", "SELECT 1 -- hi", "PRAGMA table_info(orders)"])
def test_blocks_unsafe(bad):
    with pytest.raises(UnsafeSQLError):
        validate_sql(bad)


def test_adds_limit_and_strips_fences():
    assert validate_sql("```sql\nSELECT * FROM orders\n```").endswith("LIMIT 200")


def test_cte_allowed():
    validate_sql("WITH t AS (SELECT region, SUM(revenue) r FROM orders GROUP BY 1) SELECT * FROM t")


def test_grounding_detects_invented_numbers():
    df = pd.DataFrame({"region": ["North", "South"], "total_revenue": [1000, 2500]})
    assert grounded("South leads with 2500 vs 1000.", df)
    assert not grounded("South leads with 9999.", df)


@pytest.mark.parametrize("q", ["Total revenue by region", "Return rate by product category", "Monthly revenue trend", "average csat by channel"])
def test_end_to_end(q, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = ask(q)
    assert r["ok"], r.get("error"); assert len(r["data"]) > 0 and r["summary"]
