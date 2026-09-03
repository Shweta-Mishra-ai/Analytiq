"""
engines/ml/results.py — what a training run returns.

Shapes only. Every other module in this package produces one and the
runner assembles them, so they live apart from all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    name:           str
    task:           str          # "regression" or "classification"
    cv_score:       float        # mean cross-val score
    cv_std:         float        # std of cv scores
    train_score:    float
    test_score:     float
    overfit_gap:    float        # train_score - test_score
    overfit_label:  str          # "None", "Mild", "Severe"
    metric_name:    str          # "R2", "Accuracy", "F1"
    # Regression metrics
    mae:            Optional[float] = None
    rmse:           Optional[float] = None
    # Classification metrics
    f1:             Optional[float] = None
    roc_auc:        Optional[float] = None
    # Model object (not serialized)
    model:          Any = field(default=None, repr=False)
    is_best:        bool = False


@dataclass
class FeatureImportance:
    feature:     str
    importance:  float
    rank:        int
    direction:   str    # "positive", "negative", "mixed"
    explanation: str    # plain English


@dataclass
class MLReport:
    task:               str          # "regression" or "classification"
    target_col:         str
    feature_cols:       List[str]
    n_rows_used:        int
    n_features:         int
    class_balance:      Optional[Dict] = None   # classification only
    models:             List[ModelResult] = field(default_factory=list)
    best_model:         Optional[ModelResult] = None
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    shap_values:        Optional[np.ndarray] = None
    shap_feature_names: Optional[List[str]] = None
    X_test:             Optional[pd.DataFrame] = None
    y_test:             Optional[pd.Series] = None
    y_pred:             Optional[np.ndarray] = None
    preprocessor:       Any = field(default=None, repr=False)
    label_encoders:     Dict = field(default_factory=dict)
    # {one-hot column: (source column, level)} so a feature can
    # be named for a reader and a what-if can be given a category
    # rather than a dummy.
    encoding_map:       Dict = field(default_factory=dict)
    target_encoder:     Any = field(default=None, repr=False)
    warnings:           List[str] = field(default_factory=list)
    insights:           List[str] = field(default_factory=list)
    # Whether the model beat the obvious guess. When it did not, the
    # report says so instead of quoting an accuracy that only reflects
    # the class balance, and feature importances are withheld.
    verdict:            Any = None
    leakage:            List = field(default_factory=list)
    # Columns that separate the outcome almost perfectly and could not be
    # cleared as either a leak or a real driver. Kept in the model, named
    # here, and priced by score_without_suspects.
    suspect_features:   List = field(default_factory=list)
    score_without_suspects: Any = None


# ══════════════════════════════════════════════════════════