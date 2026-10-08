"""Results -> natural-language insight, with a grounding check so the summary can't invent numbers."""
import os, re
import pandas as pd


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def template_summary(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "The query returned no rows."
    if df.shape == (1, 1):
        return f"{df.columns[0].replace('_', ' ').title()}: {df.iloc[0, 0]:,}."
    label, metric = df.columns[0], df.columns[1]
    hi, lo = df.loc[df[metric].idxmax()], df.loc[df[metric].idxmin()]
    fmt = lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return (f"Across {len(df)} groups, {hi[label]} is highest on {metric.replace('_', ' ')} ({fmt(hi[metric])}) and "
            f"{lo[label]} is lowest ({fmt(lo[metric])}).")


def grounded(summary: str, df: pd.DataFrame) -> bool:
    """Every number in the summary must appear in the result table (rounded/formatted variants allowed)."""
    allowed = set()
    for v in df.to_numpy().ravel():
        if isinstance(v, (int, float)):
            allowed |= {str(v), f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:,.0f}".replace(",", "")}
        else:
            allowed |= _numbers(str(v))
    allowed |= {str(len(df))}
    return all(n.rstrip(".") in allowed for n in _numbers(summary))


def generate_insight(question: str, df: pd.DataFrame) -> tuple[str, str]:
    """Return (summary, source): 'llm' if validated, else 'template'."""
    if os.getenv("ANTHROPIC_API_KEY") and not df.empty:
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=300,
                system="You are a business analyst. Write 2-3 sentences explaining the result table for the question. "
                       "Use ONLY numbers present in the table. Mention the biggest driver and one suggested follow-up.",
                messages=[{"role": "user", "content": f"Question: {question}\n\nResult:\n{df.head(30).to_csv(index=False)}"}])
            text = msg.content[0].text.strip()
            if grounded(text, df):
                return text, "llm"
            print("[warn] LLM summary failed grounding check; using template")
        except Exception as e:
            print(f"[warn] LLM unavailable ({e}); using template")
    return template_summary(question, df), "template"
