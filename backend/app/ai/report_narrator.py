"""
ai/report_narrator.py — column-name humanisation for report prose.

`generate_chart_narrative`, its per-chart-type stat computers, its
rule-based fallback templates, its LLM prompt builders and its
hallucination guard — roughly 470 lines — lived here and are gone. They
existed to caption an already-drawn chart from nothing but its title
string: given "YearsWithCurrManager by AgeGroup", the code below split
on " by " and searched the frame's columns for one whose name appeared
*inside* each half of the title.

That search had no word boundaries. "age" is a substring of
"yearswithcurr**mana**ger" — the tail of "manager" plus the letter
before it — so a title naming YearsWithCurrManager could match the
column Age instead, and since the search returned the first column in
the frame that matched rather than the best one, whichever of the two
happened to sit earlier in the dataframe won. The result was a chart
captioned with a different metric than the one it plotted: a bar chart
of YearsWithCurrManager narrated as being about Age throughout, "Age
across 5 Agegroup groups: '55+' leads at 57.745 while '18-25' trails at
22.407" printed directly under a picture of years-with-manager. The
line-chart half of the same module ran the same class of bug the other
way — a dataset with zero datetime columns still got a chart titled
"{column} Trend" and a narrative computed by literally comparing the
first half of the rows to the second half, "values have improved by
84.1% from the first to second half of the data," with no time
information in the file to support the word "improved" meaning
anything.

The fix was not a better parser. `app/engines/chart_exporter.py` already
knows the x and y column it just used to draw each chart — it does not
need to guess them back out of a string afterwards. Its
`generate_chart_pack_with_narratives` calls `chart_message.py`'s
functions directly, with the real columns, so the caption is generated
from the same data the picture was, and a fake "Trend" is never
manufactured for a file with no date column in the first place — see
`chart_exporter._build_chart_pack` for both fixes.

What remains here is unrelated to any of that: `clean_col` turns a raw
column name into a reader-facing label ("satisfaction_level" -> "Employee
Satisfaction Score"), used by prose that talks about a specific column
and needs it to read like language rather than a database field.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

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
