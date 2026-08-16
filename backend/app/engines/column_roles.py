"""
engines/column_roles.py — deciding what each column *is*, once.

Every engine resolved this for itself, by substring, with its own
exclusion list:

    rev_col = next((c for c in df.columns
                    if any(k in c.lower() for k in
                           ["revenue", "sales", "amount", "total"])), None)

Twenty-three separate versions of that, no two the same. The finance
engine had learned to exclude `_pct` and `rate`; the sales engine had
not, so `margin_pct` was treated as money. The sales engine had learned
nothing about `forecast_category`, so a Commit / Best Case / Pipeline
confidence band was read as the product line and the report recommended
reviewing "revenue by forecast_category for concentration and whether the
long tail justifies its resource". A shipment manifest's `cost` column
made it a finance file. `order_id` was summed as a measure.

Each of those was fixed where it was found, which left the same bug
alive in the other twenty-two places. This module is the single answer,
and it applies the discipline `domain_detect` already uses for picking a
domain:

  - **Whole words, not substrings.** `reorder_point` contains "order"
    and is not one; `discounted_price` contains "discount" and is a
    price, not a rate.
  - **Qualifiers demote.** A token that changes what the column means —
    `pct`, `forecast`, `target`, `per` — is checked before the noun it
    qualifies, because `margin_pct` is a ratio whatever else the name
    says.
  - **The dtype has to agree.** A money role needs a numeric column; a
    segment role needs a low-cardinality text one. A column called
    `revenue_band` holding "High"/"Low" is not a revenue measure.
  - **Refusing is a valid answer.** `None` produces a report that omits
    a section. A wrong match produces a fluent, confident, wrong one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from app.engines.domain_detect import tokenise
from app.engines.domains.base import is_id_column

logger = logging.getLogger(__name__)


# Tokens that change what a column means, whatever noun follows them.
# Checked first, so `margin_pct` never resolves as money and
# `forecast_category` never resolves as a product line.
RATE_TOKENS = frozenset({
    "pct", "percent", "percentage", "rate", "ratio", "share", "index",
    "per", "margin%", "score", "pp",
})
PLAN_TOKENS = frozenset({
    "forecast", "target", "quota", "budget", "plan", "planned", "goal",
    "projected", "expected", "estimate", "estimated",
})
# "category" and "segment" get used for things that are not products.
NOT_A_PRODUCT = frozenset({
    "forecast", "risk", "priority", "age", "size", "tier", "credit",
    "confidence", "probability", "stage", "lead", "customer", "grade",
    "band", "status",
})

MONEY_NOUNS = frozenset({
    "revenue", "sales", "turnover", "income", "gmv", "bookings", "billings",
    "amount", "value", "price", "total", "spend", "cost", "cogs", "expense",
    "profit", "margin", "ebitda", "opex", "capex", "fee", "charge",
    "salary", "compensation", "pay", "wage", "balance", "subtotal",
})
# Ordered: a column named `revenue` beats one named `amount` when both
# exist, because the first is unambiguous and the second is a container.
MONEY_PREFERENCE = ("revenue", "sales", "turnover", "gmv", "bookings",
                    "income", "amount", "value", "total", "price")

COST_NOUNS = frozenset({"cost", "cogs", "expense", "spend", "opex", "capex"})
PROFIT_NOUNS = frozenset({"profit", "margin", "ebitda", "ebit", "net"})
QUANTITY_NOUNS = frozenset({
    "quantity", "qty", "units", "count", "volume", "orders", "sessions",
    "visits", "clicks", "impressions", "headcount", "transactions", "seats",
})
PRODUCT_NOUNS = frozenset({"product", "sku", "item", "article", "catalogue",
                           "catalog", "category", "subcategory", "segment",
                           "line", "brand", "model"})
REGION_NOUNS = frozenset({"region", "territory", "zone", "area", "market",
                          "country", "state", "city", "district", "branch",
                          "location", "site", "store"})
PERSON_NOUNS = frozenset({"rep", "salesperson", "agent", "owner", "seller",
                          "consultant", "advisor", "manager", "employee",
                          "staff", "operator", "analyst"})
RATING_NOUNS = frozenset({"rating", "score", "csat", "nps", "satisfaction",
                          "stars"})
ATTRITION_NOUNS = frozenset({"attrition", "churn", "churned", "left",
                            "resigned", "exited", "terminated", "leaver",
                            "turnover"})

# Values that mean "this record left". Shared so a Yes/No column, a 0/1
# column and a True/False column all read the same way — four separate
# copies of this list disagreed, and two of them called `.mean()` on the
# raw column, which is only correct for the 0/1 spelling.
_LEFT_VALUES = frozenset({"yes", "y", "1", "1.0", "true", "t", "left",
                          "churned", "resigned", "exited", "terminated",
                          "attrited", "voluntary", "involuntary"})


@dataclass(frozen=True)
class Roles:
    """What the resolver found, and why.

    `reason` exists so a wrong assignment can be seen and argued with,
    rather than silently shaping every figure in the report.
    """
    money: Optional[str] = None
    cost: Optional[str] = None
    profit: Optional[str] = None
    quantity: Optional[str] = None
    rate: Optional[str] = None
    plan: Optional[str] = None
    product: Optional[str] = None
    region: Optional[str] = None
    person: Optional[str] = None
    rating: Optional[str] = None
    period: Optional[str] = None
    attrition: Optional[str] = None
    reason: Dict[str, str] = None      # role -> why that column

    def get(self, role: str) -> Optional[str]:
        return getattr(self, role, None)


def left_mask(df: pd.DataFrame):
    """(column, boolean 'this one left') for an attrition column, or None.

    Four engines found this column for themselves, with four different
    keyword lists — `_find_left_mask` and `_run_attrition` in the HR
    engine, and twice more in `insights_builder`. They also disagreed
    about how to read it: the HR engine normalised "Yes"/"No" to a
    boolean, while both copies in `insights_builder` called `.mean()` on
    the raw column, which is right for a 0/1 spelling and silently wrong
    — or an exception — for the Yes/No one that is at least as common in
    an HRIS export.
    """
    col = None
    for c in df.columns:
        if set(tokenise(c)) & ATTRITION_NOUNS:
            col = c
            break
    if col is None:
        return None
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return col, s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        # 0/1 only. A tenure-in-months column called `months_to_exit` is
        # not a flag, and averaging it as one produces a 43% attrition
        # rate out of nowhere.
        values = set(pd.to_numeric(s, errors="coerce").dropna().unique())
        if not values <= {0, 1}:
            return None
        return col, s.fillna(0).astype(float).eq(1)
    normalised = s.astype("string").str.strip().str.lower()
    mask = normalised.isin(_LEFT_VALUES)
    if not mask.any():
        return None
    return col, mask.fillna(False)


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    if pd.api.types.is_numeric_dtype(df[col]):
        return True
    # A money column exported as text with thousands separators is still
    # a money column; one holding "High"/"Low" is not.
    coerced = pd.to_numeric(df[col], errors="coerce")
    return coerced.notna().mean() >= 0.8


def _is_label(df: pd.DataFrame, col: str, max_unique: int) -> bool:
    if pd.api.types.is_numeric_dtype(df[col]):
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return False
    n = df[col].nunique(dropna=True)
    return 2 <= n <= max_unique


def _pick(df: pd.DataFrame, nouns: frozenset, *,
          numeric: bool, max_unique: int = 50,
          block: Sequence[frozenset] = (),
          prefer: Sequence[str] = (),
          exclude: Sequence[str] = ()) -> Optional[tuple]:
    """The best column matching `nouns`, or None. Returns (col, reason)."""
    candidates: List[tuple] = []
    for col in df.columns:
        if col in exclude:
            continue
        tokens = set(tokenise(col))
        if not tokens & nouns:
            continue
        blocked = set()
        for group in block:
            blocked |= tokens & group
        if blocked:
            logger.debug("%r not taken: qualified by %s", col, sorted(blocked))
            continue
        if numeric:
            if is_id_column(col, df[col]) or not _is_numeric(df, col):
                continue
        else:
            if is_id_column(col, df[col]) or not _is_label(df, col, max_unique):
                continue
        # Preference order among equally valid matches, then the earlier
        # column, so the answer does not depend on dict ordering.
        rank = next((i for i, p in enumerate(prefer) if p in tokens),
                    len(prefer))
        matched = sorted(tokens & nouns)
        candidates.append((rank, list(df.columns).index(col), col, matched))
    if not candidates:
        return None
    candidates.sort()
    _r, _i, col, matched = candidates[0]
    return col, "matched on {}".format(", ".join(matched))


def resolve(df: pd.DataFrame) -> Roles:
    """Assign each role at most one column, with the reason recorded."""
    if df is None or df.empty and not len(df.columns):
        return Roles(reason={})

    reason: Dict[str, str] = {}
    found: Dict[str, Optional[str]] = {}

    def _take(role: str, result, note: str = ""):
        if result:
            col, why = result
            found[role] = col
            reason[role] = "'{}' {}{}".format(col, why,
                                              "; " + note if note else "")
        else:
            found[role] = None

    # Rates first, so a rate column is claimed here and cannot also be
    # taken as money further down.
    _take("rate", _pick(df, RATE_TOKENS, numeric=True))
    # A plan column is a target, not an actual. Summing budget alongside
    # revenue as though both were income is how a variance becomes
    # nonsense.
    _take("plan", _pick(df, PLAN_TOKENS, numeric=True,
                        block=(RATE_TOKENS,)))

    claimed = [c for c in (found.get("rate"), found.get("plan")) if c]

    _take("profit", _pick(df, PROFIT_NOUNS, numeric=True,
                          block=(RATE_TOKENS, PLAN_TOKENS),
                          exclude=claimed))
    claimed += [c for c in (found.get("profit"),) if c]

    _take("cost", _pick(df, COST_NOUNS, numeric=True,
                        block=(RATE_TOKENS, PLAN_TOKENS),
                        exclude=claimed))
    claimed += [c for c in (found.get("cost"),) if c]

    _take("money", _pick(df, MONEY_NOUNS, numeric=True,
                         block=(RATE_TOKENS, PLAN_TOKENS),
                         prefer=MONEY_PREFERENCE,
                         exclude=claimed))
    claimed += [c for c in (found.get("money"),) if c]

    _take("quantity", _pick(df, QUANTITY_NOUNS, numeric=True,
                            block=(RATE_TOKENS,), exclude=claimed))
    _take("rating", _pick(df, RATING_NOUNS, numeric=True,
                          block=(PLAN_TOKENS,), exclude=claimed))

    _take("product", _pick(df, PRODUCT_NOUNS, numeric=False, max_unique=60,
                           block=(NOT_A_PRODUCT,)))
    _take("region", _pick(df, REGION_NOUNS, numeric=False, max_unique=60))
    _take("person", _pick(df, PERSON_NOUNS, numeric=False, max_unique=300))

    found_left = left_mask(df)
    if found_left:
        found["attrition"] = found_left[0]
        reason["attrition"] = "'{}' reads as a left/stayed flag".format(
            found_left[0])
    else:
        found["attrition"] = None

    dates = df.select_dtypes(include="datetime").columns.tolist()
    if dates:
        found["period"] = dates[0]
        reason["period"] = "'{}' is the first datetime column".format(dates[0])
    else:
        named = _pick(df, frozenset({"period", "month", "quarter", "year",
                                     "week", "date"}), numeric=False,
                      max_unique=400)
        _take("period", named)

    return Roles(reason=reason,
                 **{k: v for k, v in found.items() if k != "reason"})
