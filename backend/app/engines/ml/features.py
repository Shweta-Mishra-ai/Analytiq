"""
engines/ml/features.py — turning a frame into a matrix.

Where information is most easily lost: an encoder that imposes a false
ordering on nominal categories, an identifier left in as a feature, a
column dropped for having too many levels. Each of those is a decision
with a reason attached here rather than a default inherited from a
library.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import LabelEncoder

from app.engines.domains.base import is_id_column
from app.services.dtypes import is_text_dtype, text_columns

#: Above this many distinct values a categorical column is dropped
#: rather than one-hot encoded — a hundred dummy columns from one
#: field swamps every real signal in the matrix.
MAX_ONEHOT_LEVELS = 30


#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Prepare X, y for ML.
    - Encode categoricals
    - Handle missing values
    - Remove low-variance features
    Returns (X, y, label_encoders, encoding_map).
    """
    df = df.copy()

    # Target
    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    # Use selected features or auto-select
    if selected_features:
        available = [c for c in selected_features if c in df.columns]
        df = df[available]
    else:
        # Drop ID-like columns
        drop_cols = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                drop_cols.append(col)
                continue
            # An identifier is not a feature. EmployeeNumber was coming
            # second in the importance ranking at 22% — the model had
            # learned that low row numbers were recorded earlier, which
            # is true of the file and true of nothing else.
            if is_id_column(col, df[col]):
                drop_cols.append(col)
                continue
            # High cardinality string → drop
            if is_text_dtype(df[col]) and df[col].nunique() / max(len(df), 1) > 0.5:
                drop_cols.append(col)
        if drop_cols:
            logger.info("excluded %d non-feature column(s) from the model: %s",
                        len(drop_cols), ", ".join(map(str, drop_cols)))
        df = df.drop(columns=drop_cols)

    # One-hot, not label encoding. LabelEncoder assigns Sales=0,
    # Research=1, HR=2 — an ordering that exists nowhere in the data. A
    # linear model then reads HR as "twice Research", and a tree splits
    # on a threshold in a sequence that means nothing, so a category
    # whose effect is not monotonic in that accidental order can only be
    # captured by spending several splits on it.
    label_encoders: Dict = {}
    encoding_map: Dict = {}
    cat_cols = [c for c in text_columns(df) if c in df.columns]
    for col in cat_cols:
        levels = df[col].fillna("Unknown").astype(str)
        n_levels = int(levels.nunique())
        if n_levels > MAX_ONEHOT_LEVELS:
            # Too many levels to one-hot without swamping the feature
            # space. Named rather than silently label-encoded into a
            # false ordering.
            df = df.drop(columns=[col])
            label_encoders[col] = None
            continue
        dummies = pd.get_dummies(levels, prefix=str(col), prefix_sep="=")
        for dummy in dummies.columns:
            encoding_map[dummy] = (col, dummy.split("=", 1)[1])
        df = df.drop(columns=[col]).join(dummies.astype(float))

    # Convert datetime to numeric (days since min)
    for col in df.select_dtypes(include="datetime").columns:
        df[col] = (df[col] - df[col].min()).dt.days

    # Keep only numeric
    df = df.select_dtypes(include="number")

    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]

    # Align y with X index
    common_idx = df.index.intersection(y.dropna().index)
    df = df.loc[common_idx]
    y  = y.loc[common_idx]

    return df, y, label_encoders, encoding_map


# ══════════════════════════════════════════════════════════