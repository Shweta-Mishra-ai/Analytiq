"""
engines/ml/targets.py — what is worth predicting, and what kind of
problem it is.

Column names carry intent. "attrition" and "churn" name an outcome
somebody wants to predict; "employee_id" and "region" name an attribute
of the row. Getting that distinction wrong means offering to predict a
postcode, so the scoring here is deliberate rather than incidental.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from typing import Dict, List, Tuple

from app.engines.domains.base import is_id_column


#  TASK DETECTION
# ══════════════════════════════════════════════════════════

def detect_task(series: pd.Series) -> Tuple[str, str]:
    """
    Detect if target is regression or classification.
    Returns (task, reason).
    """
    s       = series.dropna()
    n_uniq  = s.nunique()
    dtype   = s.dtype

    # A date is not something this app predicts. It used to fall through
    # to the "continuous numeric" branch below, and the caller then did
    # arithmetic on Timestamps — which crashed the whole endpoint for any
    # dataset containing a date column, which is most of them.
    if (pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)):
        return "unsupported", "Dates are used to order and group data, not "\
                              "predicted as an outcome"

    # Boolean or binary → classification
    if n_uniq == 2:
        return "classification", "Binary target (2 unique values)"

    # Object/string → classification
    if dtype == object or str(dtype) == "str":
        if n_uniq <= 20:
            return "classification", "Categorical target ({} classes)".format(n_uniq)
        else:
            return "classification", "High-cardinality categorical ({} classes)".format(n_uniq)

    # Few unique integers → classification
    if pd.api.types.is_integer_dtype(dtype) and n_uniq <= 15:
        return "classification", "Discrete integer target ({} unique values)".format(n_uniq)

    # Continuous numeric → regression
    return "regression", "Continuous numeric target ({} unique values)".format(n_uniq)


def suggest_targets(df: pd.DataFrame) -> List[Dict]:
    """
    Suggest good target columns.
    Returns ranked list with task type and reason.
    """
    suggestions = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        # Skip ID-like columns
        if s.nunique() / max(len(s), 1) > 0.95 and len(s) > 50:
            continue

        if is_id_column(col, df[col]):
            continue

        task, reason = detect_task(s)
        if task == "unsupported":
            continue
        # Ranking used to maximise class balance, which made a seven-way
        # count of training days (evenly spread, so "balanced") outrank
        # attrition at 80/20 — the one question the dataset exists to
        # answer. What makes a good target is being an outcome, not being
        # uniform.
        score = 0.0

        if task == "regression":
            cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
            score = min(float(cv), 1.0) * 0.5
        else:
            vc = s.value_counts(normalize=True)
            n_classes = int(s.nunique())
            majority = float(vc.max())
            # A binary outcome is what most business questions are.
            score = 0.6 if n_classes == 2 else 0.35 if n_classes <= 4 else 0.1
            # Extreme imbalance is a genuine problem; ordinary imbalance
            # is what a real outcome looks like.
            if majority > 0.97:
                score *= 0.2
            elif majority > 0.9:
                score *= 0.7

        # A name that says "outcome" is the strongest signal available,
        # and the cheapest.
        if _names_an_outcome(col):
            score += 0.6
        elif _names_an_attribute(col):
            score -= 0.25

        suggestions.append({
            "column": col, "task": task,
            "reason": reason, "score": round(max(score, 0.0), 3),
            "n_unique": s.nunique(), "dtype": str(s.dtype),
        })

    return sorted(suggestions, key=lambda x: x["score"], reverse=True)


# Words that name a thing that happened, rather than a fact about a row.
_OUTCOME_TOKENS = {
    "attrition", "churn", "churned", "left", "leaver", "resigned",
    "terminated", "exited", "converted", "conversion", "default",
    "defaulted", "fraud", "fraudulent", "won", "lost", "success",
    "successful", "failed", "failure", "retained", "renewed", "survived",
    "purchased", "subscribed", "cancelled", "returned", "approved",
    "rejected", "accepted", "outcome", "status", "target", "label",
    "response", "responded", "click", "clicked", "readmitted", "escalated",
}

# Words that name a property of the row, which is a feature.
_ATTRIBUTE_TOKENS = {
    "count", "number", "times", "total", "sum", "age", "year", "years",
    "month", "day", "date", "rating", "score", "level", "band", "grade",
    "code", "type", "category", "name", "region", "city", "country",
    "department", "role", "title", "gender", "hours", "distance",
}


def _token_set(col) -> set:
    import re as _re
    spaced = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col)).lower()
    return {t for t in _re.split(r"[^a-z0-9]+", spaced) if t}


def _names_an_outcome(col) -> bool:
    return bool(_token_set(col) & _OUTCOME_TOKENS)


def _names_an_attribute(col) -> bool:
    return bool(_token_set(col) & _ATTRIBUTE_TOKENS)


# ══════════════════════════════════════════════════════════