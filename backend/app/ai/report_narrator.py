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


def clean_col(col: str) -> str:
    low = col.lower().strip()
    if low in _COL_MAP:
        return _COL_MAP[low]
    # Try prompt_builder if available
    try:
        from app.ai.prompt_builder import translate_column_name
        return translate_column_name(col)
    except Exception:
        pass
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
#  STAT COMPUTERS  (always work, no external deps)
# ══════════════════════════════════════════════════════════

def _bar_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp = df.groupby(x)[y].mean().sort_values(ascending=False)
        avg = float(df[y].mean())
        return {
            "ok": True, "chart": "bar",
            "x_col": x, "y_col": y,
            "metric_label":    clean_col(y),
            "dimension_label": clean_col(x),
            "top":     str(grp.index[0]),
            "top_val": round(float(grp.iloc[0]), 3),
            "worst":     str(grp.index[-1]),
            "worst_val": round(float(grp.iloc[-1]), 3),
            "gap_pct": round(abs(float(grp.iloc[0]) - float(grp.iloc[-1])) /
                             max(abs(float(grp.iloc[-1])), 0.001) * 100, 1),
            "org_avg":   round(avg, 3),
            "above_avg": int((grp > avg).sum()),
            "n_groups":  len(grp),
            "all_values": {str(k): round(float(v), 3) for k, v in grp.items()},
        }
    except Exception as e:
        return {"ok": False, "metric_label": clean_col(y),
                "dimension_label": clean_col(x), "error": str(e)}


def _hist_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s    = pd.to_numeric(df[col], errors="coerce").dropna()
        s    = s[np.isfinite(s)]
        skew = float(s.skew())
        return {
            "ok": True, "chart": "histogram",
            "col": col, "metric_label": clean_col(col),
            "mean":   round(float(s.mean()),   3),
            "median": round(float(s.median()), 3),
            "std":    round(float(s.std()),    3),
            "min":    round(float(s.min()),    3),
            "max":    round(float(s.max()),    3),
            "q1":     round(float(s.quantile(0.25)), 3),
            "q3":     round(float(s.quantile(0.75)), 3),
            "skew":   round(skew, 2),
            "shape":  ("right-skewed" if skew > 0.5
                       else "left-skewed" if skew < -0.5
                       else "symmetric"),
            "use_stat": "median" if abs(skew) > 0.5 else "mean",
            "use_val":  round(float(s.median()) if abs(skew) > 0.5
                              else float(s.mean()), 3),
        }
    except Exception as e:
        return {"ok": False, "metric_label": clean_col(col), "error": str(e)}


def _pie_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    try:
        grp   = df.groupby(x)[y].mean().sort_values(ascending=False)
        total = grp.sum()
        return {
            "ok": True, "chart": "pie",
            "x_col": x, "y_col": y,
            "metric_label":    clean_col(y),
            "dimension_label": clean_col(x),
            "n_segments": len(grp),
            "top_seg":  str(grp.index[0]),
            "top_pct":  round(grp.max() / total * 100, 1),
            "top2_pct": round(grp.nlargest(2).sum() / total * 100, 1),
            "shares":   {str(k): round(v / total * 100, 1) for k, v in grp.items()},
            "balanced": bool(grp.max() / total * 100 < 40),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _trend_stats(df: pd.DataFrame, col: str) -> dict:
    try:
        s   = pd.to_numeric(df[col], errors="coerce").dropna()
        s   = s[np.isfinite(s)]
        cv  = s.std() / abs(s.mean()) * 100 if s.mean() != 0 else 0
        mid = len(s) // 2
        f   = float(s.iloc[:mid].mean())
        sc  = float(s.iloc[mid:].mean())
        pct = (sc - f) / max(abs(f), 0.001) * 100
        return {
            "ok": True, "chart": "trend",
            "col": col, "metric_label": clean_col(col),
            "mean":      round(float(s.mean()), 3),
            "min":       round(float(s.min()), 3),
            "max":       round(float(s.max()), 3),
            "cv":        round(cv, 1),
            "first_half": round(f, 3),
            "sec_half":   round(sc, 3),
            "trend_pct":  round(pct, 1),
            "trend_dir":  ("improved" if pct > 2
                           else "declined" if pct < -2
                           else "stable"),
        }
    except Exception as e:
        return {"ok": False, "metric_label": clean_col(col), "error": str(e)}


def _corr_stats(df: pd.DataFrame) -> list:
    try:
        num  = df.select_dtypes(include="number").columns.tolist()[:8]
        corr = df[num].corr(method="spearman")
        pairs = []
        for i in range(len(num)):
            for j in range(i + 1, len(num)):
                a, b = num[i], num[j]
                r    = float(corr.loc[a, b])
                if abs(r) >= 0.15:
                    pairs.append((a, b, r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs
    except Exception:
        return []


# ══════════════════════════════════════════════════════════
#  RULE-BASED FALLBACKS  (guaranteed output, no LLM needed)
# ══════════════════════════════════════════════════════════

def _fb_bar(s: dict) -> str:
    if not s.get("ok"):
        return f"Analysis of {s.get('metric_label','metric')} by group reveals performance patterns."
    return (
        f"{s['metric_label']} across {s['n_groups']} {s['dimension_label']} groups: "
        f"'{s['top']}' leads at {s['top_val']} while '{s['worst']}' trails "
        f"at {s['worst_val']} — a {s['gap_pct']:.0f}% gap requiring attention. "
        f"{s['above_avg']} of {s['n_groups']} groups exceed the organisation "
        f"average of {s['org_avg']}. "
        f"Strategic Action: Conduct root-cause review in '{s['worst']}' and "
        f"replicate '{s['top']}' practices to close the {s['gap_pct']:.0f}% gap."
    )


def _fb_hist(s: dict) -> str:
    if not s.get("ok"):
        return f"Distribution of {s.get('metric_label','metric')} reveals key patterns."
    return (
        f"{s['metric_label']} typical value is {s['use_val']} "
        f"(range: {s['min']}–{s['max']}). "
        f"{'Skewed distribution — use ' + s['use_stat'] + ' (' + str(s['use_val']) + ') for accurate reporting.' if s['shape'] != 'symmetric' else 'Symmetric distribution — mean is a reliable central measure.'} "
        f"Employees below {s['q1']} (bottom 25%) represent the highest-risk group. "
        f"Strategic Action: Focus retention interventions on employees below "
        f"{s['q1']} to address the most at-risk segment first."
    )


def _fb_pie(s: dict) -> str:
    if not s.get("ok"):
        return "Composition chart reveals segment distribution patterns."
    return (
        f"{s['metric_label']} composition across {s['n_segments']} "
        f"{s['dimension_label']} segments: "
        f"'{s['top_seg']}' holds {s['top_pct']}%, "
        f"top 2 combined: {s['top2_pct']}%. "
        f"{'Concentration risk — business is over-reliant on a dominant segment.' if not s['balanced'] else 'Well-balanced distribution — no single segment dominates.'} "
        f"Strategic Action: "
        f"{'Diversify activity away from the dominant segment to reduce concentration risk.' if not s['balanced'] else 'Maintain this balance and monitor quarterly for emerging concentration.'}"
    )


def _fb_trend(s: dict) -> str:
    if not s.get("ok"):
        return f"Trend analysis of {s.get('metric_label','metric')} shows performance over time."
    return (
        f"{s['metric_label']} trend: overall average {s['mean']} "
        f"(range {s['min']}–{s['max']}). "
        f"Values have {s['trend_dir']} by {abs(s['trend_pct']):.1f}% "
        f"from the first to second half of the data — "
        f"{'a positive signal worth sustaining.' if s['trend_dir'] == 'improved' else 'a declining trend requiring immediate investigation.' if s['trend_dir'] == 'declined' else 'a consistently stable pattern.'} "
        f"Variability across the range is {s['cv']}% "
        f"({'high — inconsistent performance' if s['cv'] > 30 else 'moderate' if s['cv'] > 15 else 'stable'}). "
        f"Strategic Action: "
        f"{'Identify and address the root causes of the decline before the next review cycle.' if s['trend_dir'] == 'declined' else 'Continue current practices and monitor monthly for early warning signals.'}"
    )


def _fb_corr(pairs: list, df: pd.DataFrame) -> str:
    if not pairs:
        return (
            "The correlation matrix shows no meaningful relationships between "
            "variables (all |r| < 0.15). Each variable operates independently — "
            "single-variable interventions are unlikely to create cross-metric spillover. "
            "Strategic Action: Design targeted interventions per metric rather than "
            "expecting compound effects."
        )
    a, b, r = pairs[0]
    direction = "positive" if r > 0 else "negative"
    meaning   = (f"higher {clean_col(a)} tends to occur alongside higher {clean_col(b)}"
                 if r > 0 else
                 f"as {clean_col(a)} increases, {clean_col(b)} tends to decrease")
    second = ""
    if len(pairs) > 1:
        a2, b2, r2 = pairs[1]
        second = (f" Second strongest: {clean_col(a2)} and {clean_col(b2)} "
                  f"(r={r2:.2f}, r²={r2**2:.2f}).")
    return (
        f"The correlation matrix reveals {len(pairs)} meaningful relationships "
        f"across the dataset. "
        f"Strongest: {clean_col(a)} and {clean_col(b)} (r={r:.2f}, {direction}) — "
        f"{meaning}. "
        f"r²={r**2:.2f} means only {r**2*100:.0f}% of variance is shared — "
        f"this is association, not causation.{second} "
        f"Strategic Action: Test the {clean_col(a)}–{clean_col(b)} relationship "
        f"through a controlled analysis before committing to a causal explanation."
    )


# ══════════════════════════════════════════════════════════
#  PROMPT BUILDERS  (lazy imports — won't crash if pb missing)
# ══════════════════════════════════════════════════════════

def _build_chart_prompt(ctype: str, stats: dict, domain: str) -> str:
    """
    Build chart prompt using prompt_builder if available.
    Returns "" if prompt_builder unavailable — caller uses fallback.
    """
    domain_lbl = _DOMAIN_LABELS.get(domain, "Business Analytics")
    try:
        from app.ai.prompt_builder import (
            BAR_CHART_PROMPT, PIE_CHART_PROMPT,
            LINE_CHART_PROMPT, DISTRIBUTION_CHART_PROMPT,
        )
    except ImportError:
        return ""

    try:
        if ctype == "bar":
            chart_data = (
                f"Top: '{stats['top']}' = {stats['top_val']} | "
                f"Worst: '{stats['worst']}' = {stats['worst_val']} | "
                f"Avg: {stats['org_avg']} | Gap: {stats['gap_pct']:.0f}% | "
                f"Above avg: {stats['above_avg']}/{stats['n_groups']} | "
                f"All: {stats['all_values']}"
            )
            return BAR_CHART_PROMPT.format(
                domain=domain_lbl,
                metric_label=stats["metric_label"],
                dimension_label=stats["dimension_label"],
                raw_metric=stats["y_col"],
                raw_dimension=stats["x_col"],
                chart_data=chart_data,
            )
        elif ctype == "pie":
            chart_data = (
                f"Shares: {stats['shares']} | "
                f"Largest: '{stats['top_seg']}' at {stats['top_pct']}% | "
                f"Top 2: {stats['top2_pct']}% | "
                f"{'CONCENTRATED' if not stats['balanced'] else 'BALANCED'}"
            )
            return PIE_CHART_PROMPT.format(
                domain=domain_lbl,
                metric_label=stats["metric_label"],
                dimension_label=stats["dimension_label"],
                raw_metric=stats["y_col"],
                raw_dimension=stats["x_col"],
                chart_data=chart_data,
            )
        elif ctype == "trend":
            chart_data = (
                f"Mean: {stats['mean']} | Range: {stats['min']}–{stats['max']} | "
                f"First half: {stats['first_half']} | Second half: {stats['sec_half']} | "
                f"Change: {stats['trend_pct']:+.1f}% ({stats['trend_dir']}) | "
                f"CV: {stats['cv']}%"
            )
            return LINE_CHART_PROMPT.format(
                domain=domain_lbl,
                metric_label=stats["metric_label"],
                raw_metric=stats.get("col", "metric"),
                chart_data=chart_data,
            )
        elif ctype == "hist":
            return DISTRIBUTION_CHART_PROMPT.format(
                domain=domain_lbl,
                metric_label=stats["metric_label"],
                mean_val=stats["mean"],
                median_val=stats["median"],
                min_val=stats["min"],
                max_val=stats["max"],
            )
    except Exception as e:
        logger.warning(f"Prompt format error [{ctype}]: {e}")

    return ""


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
    cat_cols = df.select_dtypes(include="object").columns.tolist()
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

def generate_chart_narrative(
    df:           pd.DataFrame,
    chart_title:  str,
    groq_api_key: str = "",
    domain:       str = "general",
) -> str:
    """
    Generate chart narrative — GUARANTEED non-empty output.
    Flow: compute stats → try LLM with domain prompt → validate → fallback.
    Never throws — all exceptions handled internally.
    """
    try:
        title = chart_title.lower()
        num   = df.select_dtypes(include="number").columns.tolist()
        cat   = df.select_dtypes(include="object").columns.tolist()

        # ── Correlation ───────────────────────────────────
        if "correlation" in title or "heatmap" in title:
            pairs = _corr_stats(df)
            return _fb_corr(pairs, df)   # always rule-based for correlation

        # ── Histogram / Distribution ──────────────────────
        elif "distribution" in title or "histogram" in title:
            col = next((c for c in num if c.lower() in title),
                       num[0] if num else None)
            if not col:
                return f"Distribution analysis of {title}."
            s       = _hist_stats(df, col)
            prompt  = _build_chart_prompt("hist", s, domain)
            raw     = _llm_call(prompt, groq_api_key, "chart_analysis", 280)
            if raw:
                cleaned = _clean_output(raw)
                if not _is_hallucinated(cleaned, df) and len(cleaned) > 50:
                    return cleaned
            return _fb_hist(s)

        # ── Pie / Share ───────────────────────────────────
        elif "pie" in title or "share" in title:
            parts = title.split(" by ")
            x = next((c for c in cat
                       if c.lower() in (parts[1] if len(parts) > 1 else "")),
                      cat[0] if cat else None)
            y = next((c for c in num if c.lower() in parts[0]),
                      num[0] if num else None)
            if not x or not y:
                return "Pie chart composition analysis."
            s       = _pie_stats(df, x, y)
            prompt  = _build_chart_prompt("pie", s, domain)
            raw     = _llm_call(prompt, groq_api_key, "chart_analysis", 280)
            if raw:
                cleaned = _clean_output(raw)
                if not _is_hallucinated(cleaned, df) and len(cleaned) > 50:
                    return cleaned
            return _fb_pie(s)

        # ── Trend / Line ──────────────────────────────────
        elif "trend" in title or "line" in title:
            col = next((c for c in num
                         if c.lower().replace("_","") in
                         title.replace("_","").replace(" ","")),
                        num[0] if num else None)
            if not col:
                return "Trend analysis chart."
            s       = _trend_stats(df, col)
            prompt  = _build_chart_prompt("trend", s, domain)
            raw     = _llm_call(prompt, groq_api_key, "chart_analysis", 280)
            if raw:
                cleaned = _clean_output(raw)
                if not _is_hallucinated(cleaned, df) and len(cleaned) > 50:
                    return cleaned
            return _fb_trend(s)

        # ── Bar (X by Y) ──────────────────────────────────
        elif " by " in title:
            parts = title.replace("avg ","").replace("total ","").split(" by ")
            y = next((c for c in num
                       if c.lower().replace("_","") in parts[0].replace(" ","")),
                      num[0] if num else None)
            x = next((c for c in cat
                       if c.lower() in (parts[1] if len(parts) > 1 else "")),
                      cat[0] if cat else None)
            if not y or not x:
                return "Bar chart analysis."
            s       = _bar_stats(df, x, y)
            prompt  = _build_chart_prompt("bar", s, domain)
            raw     = _llm_call(prompt, groq_api_key, "chart_analysis", 280)
            if raw:
                cleaned = _clean_output(raw)
                if not _is_hallucinated(cleaned, df) and len(cleaned) > 50:
                    return cleaned
            return _fb_bar(s)

        # ── Generic ───────────────────────────────────────
        else:
            if cat and num:
                s = _bar_stats(df, cat[0], num[0])
                return _fb_bar(s)
            elif num:
                s = _hist_stats(df, num[0])
                return _fb_hist(s)
            else:
                return "Chart analysis from dataset."

    except Exception as e:
        logger.error(f"generate_chart_narrative failed for '{chart_title}': {e}")
        # Last resort — never return generic text
        try:
            num = df.select_dtypes(include="number").columns.tolist()
            cat = df.select_dtypes(include="object").columns.tolist()
            if cat and num:
                s = _bar_stats(df, cat[0], num[0])
                return _fb_bar(s)
            elif num:
                s = _hist_stats(df, num[0])
                return _fb_hist(s)
        except Exception:
            pass
        return (f"Analysis of {chart_title}: "
                f"dataset contains {len(df):,} records across "
                f"{len(df.columns)} variables.")


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
