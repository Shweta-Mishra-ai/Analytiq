"""
ai/report_narrator.py — Analytiq FINAL v9
KEY FIX: prompt_builder imports moved INSIDE functions (lazy imports).
If prompt_builder unavailable → rule-based fallbacks kick in automatically.
Charts will NEVER show "Chart generated from dataset analysis." again.

Uses prompt_builder.py domain prompts when available:
  HR   → HR_EXECUTIVE_PROMPT + HR_INSIGHT_PROMPT
  Sales → SALES_EXECUTIVE_PROMPT + SALES_INSIGHT_PROMPT
  Ecom  → ECOMMERCE_EXECUTIVE_PROMPT + ECOMMERCE_INSIGHT_PROMPT
  Finance → FINANCE_EXECUTIVE_PROMPT

Anti-hallucination:
  Python computes stats → LLM narrates → validated → fallback if bad
"""

from __future__ import annotations
import re
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hallucination phrases ─────────────────────────────────
FAKE_PHRASES = [
    "customer satisfaction and sales revenue",
    "sales revenue tends to increase",
    "marketing spend and website traffic",
    "top-line growth", "sales targeted",
    "customer-centric initiatives",
    "as customer satisfaction increases",
    "net promoter score", "website traffic", "marketing spend",
]

# ── Inline column map (no external dependency needed) ─────
_COL_MAP = {
    "satisfaction_level":    "Employee Satisfaction Score",
    "last_evaluation":       "Last Performance Evaluation",
    "number_project":        "Number of Active Projects",
    "average_montly_hours":  "Average Monthly Hours Worked",
    "average_monthly_hours": "Average Monthly Hours Worked",
    "time_spend_company":    "Employee Tenure (Years)",
    "work_accident":         "Work Accident Rate",
    "left":                  "Employee Attrition",
    "attrition":             "Employee Attrition Rate",
    "promotion_last_5years": "Recent Promotions (Last 5 Years)",
    "dept":                  "Department",
    "department":            "Department",
    "salary":                "Salary Band",
    "discounted_price":      "Selling Price",
    "actual_price":          "Original Price (MRP)",
    "discount_percentage":   "Discount Percentage",
    "rating_count":          "Number of Customer Reviews",
    "rating":                "Customer Rating",
    "product_name":          "Product Name",
    "category":              "Product Category",
    "revenue":               "Revenue",
    "sales":                 "Sales Amount",
    "target":                "Sales Target",
    "profit":                "Profit",
    "margin":                "Profit Margin",
    "region":                "Sales Region",
}

_DOMAIN_LABELS = {
    "hr": "HR", "ecommerce": "eCommerce",
    "sales": "Sales", "finance": "Finance", "general": "Business Analytics",
}


from app.services.dtypes import text_columns


def clean_col(col: str) -> str:
    low = col.lower().strip()
    if low in _COL_MAP:
        return _COL_MAP[low]
    # Try prompt_builder if available
    try:
        from app.ai.prompt_builder import translate_column_name
        return translate_column_name(col)
    except Exception:
        logger.debug("clean_col: suppressed exception", exc_info=True)
    return " ".join(w.capitalize()
                    for w in col.replace("_", " ").replace("montly", "Monthly").split())


def _is_hallucinated(text: str, df: pd.DataFrame) -> bool:
    tl = text.lower()
    if any(p in tl for p in FAKE_PHRASES):
        return True
    real = [c.lower().replace("_", "") for c in df.columns]
    for inv in ["salesrevenue", "customerrevenue", "websitetraffic",
                "marketingspend", "operationalcost", "churnrate"]:
        if inv in tl.replace(" ", "").replace("_", ""):
            if not any(inv in r for r in real):
                return True
    return False


def _clean_output(text: str) -> str:
    text = re.sub(r"\bSales Targeted\b",       "targeted",      text, flags=re.IGNORECASE)
    text = re.sub(r"\bSales Representative\b", "representative", text, flags=re.IGNORECASE)
    for raw, clean in _COL_MAP.items():
        text = text.replace(f"'{raw}'", clean).replace(f'"{raw}"', clean)
    return text.strip()


# ══════════════════════════════════════════════════════════
#  PROMPT BUILDERS  (lazy imports — won't crash if pb missing)
# ══════════════════════════════════════════════════════════

def _build_exec_prompt(df: pd.DataFrame, domain: str) -> str:
    """Build executive summary prompt using prompt_builder if available."""
    try:
        from app.ai.prompt_builder import (
            HR_EXECUTIVE_PROMPT, ECOMMERCE_EXECUTIVE_PROMPT,
            SALES_EXECUTIVE_PROMPT, FINANCE_EXECUTIVE_PROMPT,
        )
        prompts = {
            "hr": HR_EXECUTIVE_PROMPT,
            "ecommerce": ECOMMERCE_EXECUTIVE_PROMPT,
            "sales": SALES_EXECUTIVE_PROMPT,
            "finance": FINANCE_EXECUTIVE_PROMPT,
        }
        template = prompts.get(domain, HR_EXECUTIVE_PROMPT)
    except ImportError:
        return ""

    summary = _build_raw_summary(df, domain)
    try:
        return template.format(raw_data_summary=summary)
    except Exception:
        return ""


def _build_raw_summary(df: pd.DataFrame, domain: str) -> str:
    """Rich pre-computed stats for executive prompt injection."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = text_columns(df)
    lines    = [
        f"Dataset: {len(df):,} rows, {len(df.columns)} columns — {domain.upper()} domain",
        "",
        "KEY METRICS:",
    ]
    for col in num_cols[:7]:
        try:
            s    = pd.to_numeric(df[col], errors="coerce").dropna()
            s    = s[np.isfinite(s)]
            skew = float(s.skew())
            lbl  = clean_col(col)
            use  = "Median" if abs(skew) > 0.5 else "Mean"
            val  = round(float(s.median()) if abs(skew) > 0.5 else float(s.mean()), 3)
            lines.append(
                f"• {lbl}: {use}={val} | "
                f"Range={round(float(s.min()),2)}–{round(float(s.max()),2)}"
                + (" [SKEWED — use median]" if abs(skew) > 0.5 else "")
            )
        except Exception:
            logger.debug("_build_raw_summary: suppressed exception", exc_info=True)
            continue

    lines.append("")
    lines.append("CATEGORICAL BREAKDOWN:")
    for col in cat_cols[:4]:
        try:
            vc = df[col].value_counts(normalize=True).head(5)
            lines.append(
                f"• {clean_col(col)}: " +
                " | ".join([f"{k}: {v*100:.0f}%" for k, v in vc.items()])
            )
        except Exception:
            logger.debug("_build_raw_summary: suppressed exception", exc_info=True)
            continue

    # HR-specific attrition
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned","exited")), None)
    if atr_col:
        rate   = float(df[atr_col].mean()) * 100
        n_left = int(df[atr_col].sum())
        lines.extend(["", "ATTRITION:",
            f"• Rate: {rate:.1f}% ({n_left:,} left of {len(df):,})",
            f"• Benchmark: 10–15%. Gap: {max(0,rate-15):.1f}pp above",
        ])
        dept_col = next((c for c in cat_cols
                         if c.lower() in ("department","dept")), None)
        if dept_col:
            atr_d = df.groupby(dept_col)[atr_col].mean() * 100
            lines.append(
                f"• By department: "
                + " | ".join([f"{k}: {v:.0f}%" for k,v in
                               atr_d.sort_values(ascending=False).items()])
            )
        sal_col = next((c for c in cat_cols if c.lower() == "salary"), None)
        if sal_col:
            atr_s = df.groupby(sal_col)[atr_col].mean() * 100
            lines.append(
                "• By salary band: " +
                " | ".join([f"{k}: {v:.0f}%" for k,v in atr_s.items()])
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  LLM CALLER
# ══════════════════════════════════════════════════════════

def _llm_call(prompt: str, groq_api_key: str = "",
              task: str = "chart_analysis",
              max_tokens: int = 350) -> Optional[str]:
    """Call LLM. Returns None silently on any failure."""
    if not prompt:
        return None
    try:
        from app.ai.llm_client import get_client
        client = get_client(groq_api_key)
        return client.chat_task(
            system     = ("Follow the exact format and rules specified. "
                          "Only cite numbers from the provided data. "
                          "Never invent figures or external company names."),
            user       = prompt,
            task       = task,
            max_tokens = max_tokens,
        )
    except Exception as e:
        logger.warning(f"LLM call failed [{task}]: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_executive_summary(
    df:           pd.DataFrame,
    domain:       str = "general",
    story_report  = None,
    groq_api_key: str = "",
) -> str:
    """Executive summary — domain-aware prompt, guaranteed non-empty."""
    try:
        prompt = _build_exec_prompt(df, domain)
        if prompt:
            raw = _llm_call(prompt, groq_api_key,
                            task="executive_summary", max_tokens=700)
            if raw:
                cleaned = _clean_output(raw)
                if not _is_hallucinated(cleaned, df) and len(cleaned) > 80:
                    return cleaned
    except Exception as e:
        logger.warning(f"Executive summary LLM failed: {e}")

    # Rule-based fallback
    atr_col = next((c for c in df.columns
                    if c.lower() in ("left","attrition","churned","exited")), None)
    parts = [
        f"This {domain.upper()} dataset ({len(df):,} records) "
        "reveals critical patterns requiring executive attention."
    ]
    if atr_col:
        rate   = float(df[atr_col].mean()) * 100
        n_left = int(df[atr_col].sum())
        parts.append(
            f"Attrition is {rate:.1f}% ({n_left:,} employees left) — "
            f"{'above' if rate > 15 else 'at'} the healthy 10–15% benchmark."
        )
    parts.append(
        "Immediate action on the critical findings below is required "
        "to prevent further financial and operational impact."
    )
    return " ".join(parts)
