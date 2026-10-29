"""
ml_engine.py — model training, held-out evaluation, feature effects and
what-if prediction.
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
)

from app.engines.chart_exporter import fmt
from app.engines.domains.base import is_id_column
from app.services.dtypes import is_text_dtype
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    # What the models were ranked on: R² for regression, log-loss (negated)
    # for classification.
    cv_pick:        float = 0.0


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
    X_test:             Optional[pd.DataFrame] = None
    y_test:             Optional[pd.Series] = None
    y_pred:             Optional[np.ndarray] = None
    target_encoder:     Any = field(default=None, repr=False)
    # Columns trained as categories (one-hot inside the pipeline).
    categorical_cols:   List[str] = field(default_factory=list)
    # 10th and 90th percentile of held-out errors (regression): the range
    # a what-if prediction is quoted with.
    residual_band:      Optional[Tuple[float, float]] = None
    # Datetime features are trained as days since this origin; a what-if
    # input date has to be converted the same way.
    datetime_origins:   Dict = field(default_factory=dict)
    # One entry per feature describing what a what-if input may be: its
    # kind (category / date / number), the allowed values or range, and a
    # typical value from the training data to start from.
    feature_inputs:     List[Dict] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)
    insights:           List[str] = field(default_factory=list)
    # Why the task was chosen ("Continuous numeric target (…)"): shown in
    # the app, not printed in the report.
    task_reason:        str = ""
    # Set when the best model is no better than a trivial guess. Feature
    # importances always sum to 100%, so they still rank — noise.
    no_signal:          str = ""


# ══════════════════════════════════════════════════════════
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

    # Boolean or binary → classification
    if n_uniq == 2:
        return "classification", "Binary target (2 unique values)"

    # Text → classification. `dtype == object or str(dtype) == "str"` missed
    # pandas' "string" dtype, which then fell through to regression on text.
    if is_text_dtype(dtype):
        if n_uniq <= 20:
            return "classification", "Categorical target ({} classes)".format(n_uniq)
        else:
            return "classification", "High-cardinality categorical ({} classes)".format(n_uniq)

    # Few unique integers → classification
    if pd.api.types.is_integer_dtype(dtype) and n_uniq <= 15:
        return "classification", "Discrete integer target ({} unique values)".format(n_uniq)

    # Continuous numeric → regression
    return "regression", "Continuous numeric target ({} unique values)".format(n_uniq)


# A classification target with more classes than this is a lookup, not a
# prediction: 40 product types from two numeric columns took 37 seconds to
# train and scored 2%.
MAX_CLASSES = 20


class TargetError(ValueError):
    """A target this pipeline cannot model, explained for the user."""


def _target_kind(col: str, s: pd.Series, task: str) -> float:
    """How much a column reads as something worth predicting.

    Outcomes (attrition, churn, default) and money or volume measures come
    first; a grouping column such as department or region is a way to slice
    the data, not an outcome, and was the first suggestion for HR data.
    """
    from app.engines.chart_exporter import _ADDITIVE, _GROUP_WORDS, _MONEY, _words
    from app.engines.predictive import _BINARY_TARGET_NAMES, _EVENT_WORDS
    words = _words(col)
    bonus = 0.0
    if words & set(_BINARY_TARGET_NAMES) or (
            task == "classification" and s.nunique() == 2
            and {str(v).lower() for v in s.unique()} & (_EVENT_WORDS - {"1", "1.0", "y", "t"})):
        bonus += 1.0
    if task == "regression" and words & (_MONEY | _ADDITIVE):
        bonus += 0.6
    if words & _GROUP_WORDS:
        bonus -= 0.5
    return bonus


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
        # A date is not a target this pipeline can model (it would be
        # "regression" on Timestamps, and abs() of a Timestamp raises).
        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        # Skip identifiers. "Nearly every value distinct" is not the test:
        # a continuous measure — salary, revenue, price — is all-distinct
        # too, and that rule hid exactly the columns most worth predicting.
        if is_id_column(col, s):
            continue
        # Free text with a different value on most rows is not a class label.
        if is_text_dtype(s) and s.nunique() / max(len(s), 1) > 0.5:
            continue

        task, reason = detect_task(s)
        if task == "classification" and s.nunique() > MAX_CLASSES:
            continue

        if task == "regression":
            cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
            score = min(float(cv), 1.0)
        else:
            # Prefer balanced classes
            vc = s.value_counts(normalize=True)
            score = float(1 - vc.max())
        score += _target_kind(str(col), s, task)

        suggestions.append({
            "column": col, "task": task,
            "reason": reason, "score": round(score, 3),
            "n_unique": s.nunique(), "dtype": str(s.dtype),
        })

    return sorted(suggestions, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def _as_category(s: pd.Series) -> pd.Series:
    """Text, flag or category values as plain strings, missing kept missing."""
    out = pd.Series(np.nan, index=s.index, dtype=object)
    filled = s.notna()
    out[filled] = s[filled].astype(str).astype(object)
    return out


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict]:
    """
    X and y for training, in the user's own terms.

    Categories stay as their values ("Sales", "HR") and are one-hot encoded
    inside the model pipeline. They used to be label-encoded here, which
    handed the linear models an order (HR=1 < Sales=2) that does not exist,
    and made a blank what-if input the median code — an arbitrary category.
    Dates become days since the earliest date.

    Returns (X, y, categorical columns, datetime origins).
    """
    df = df.copy()

    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    if selected_features:
        df = df[[c for c in selected_features if c in df.columns]]
    else:
        drop_cols = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                drop_cols.append(col)
                continue
            # Free text that differs on most rows is a label, not a feature.
            if is_text_dtype(df[col]) and s.nunique() / max(len(df), 1) > 0.5:
                drop_cols.append(col)
                continue
            # Identifiers (employee_id, order_no, a 1..n row number) carry no
            # information about the outcome, but a model will happily fit
            # them — and on data sorted by date they leak time.
            if is_id_column(col, df[col]):
                drop_cols.append(col)
        df = df.drop(columns=drop_cols)

    cat_cols, datetime_origins = [], {}
    for col in list(df.columns):
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            origin = s.min()
            datetime_origins[col] = origin
            df[col] = (s - origin).dt.days.astype(float)
        elif (pd.api.types.is_bool_dtype(s) or is_text_dtype(s)
              or isinstance(s.dtype, pd.CategoricalDtype)):
            df[col] = _as_category(s)
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            df = df.drop(columns=[col])

    # Constant columns teach nothing.
    df = df.loc[:, df.nunique(dropna=True) > 1]
    cat_cols = [c for c in cat_cols if c in df.columns]
    datetime_origins = {c: o for c, o in datetime_origins.items() if c in df.columns}

    keep = y.notna()
    return df.loc[keep], y.loc[keep], cat_cols, datetime_origins


# ══════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════

def _get_models(task: str, n_classes: int = 2, n_rows: int = 0) -> List[Tuple[str, Any]]:
    """Return list of (name, model) tuples for given task.

    Histogram gradient boosting replaces sklearn's classic gradient
    boosting: the same method, an order of magnitude faster on large data,
    which is where training used to hang.

    Each forest tree sees at most 20,000 rows, and a regression forest
    considers half the features at each split rather than all of them: on
    60,000 rows the forest alone took 56 of 74 seconds.
    """
    per_tree = min(1.0, 20_000 / n_rows) if n_rows else None
    if task == "regression":
        models = [
            ("Ridge Regression",    Ridge(alpha=1.0)),
            ("Random Forest",       RandomForestRegressor(
                                        n_estimators=150, min_samples_leaf=5,
                                        max_features=0.5, max_samples=per_tree,
                                        random_state=42, n_jobs=-1)),
            ("Gradient Boosting",   HistGradientBoostingRegressor(
                                        max_iter=300, early_stopping=True,
                                        random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, n_jobs=-1)))
    else:
        models = [
            ("Logistic Regression", LogisticRegression(
                                        max_iter=2000, random_state=42)),
            # Leaves of at least five rows: with two, the forest's
            # probabilities on noise ran to 99% and it won on F1.
            ("Random Forest",       RandomForestClassifier(
                                        n_estimators=150, min_samples_leaf=5,
                                        max_samples=per_tree,
                                        random_state=42, n_jobs=-1)),
            # Early stopping: a 9-class target took 18 s without it, 1 s with.
            ("Gradient Boosting",   HistGradientBoostingClassifier(
                                        max_iter=300, early_stopping=True,
                                        random_state=42)),
        ]
        if XGBOOST_AVAILABLE and n_classes <= 10:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, n_jobs=-1)))
    return models


def _make_pipeline(model, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """Numbers imputed with the median and scaled; categories filled with
    their most common value and one-hot encoded, with rare values (and
    values never seen in training) grouped together."""
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=5, max_categories=30,
                                      sparse_output=False)),
        ]), cat_cols))
    return Pipeline([
        ("prep",  ColumnTransformer(parts)),
        ("model", model),
    ])


def _evaluate_regression(y_true, y_pred) -> Dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def _evaluate_classification(y_true, y_pred, y_prob=None) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="weighted")
            auc = round(auc, 4)
        except Exception:
            logger.debug("_evaluate_classification: suppressed exception", exc_info=True)
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "roc_auc": auc}


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    target_encoder: Optional[LabelEncoder] = None,
    cat_cols: Optional[List[str]] = None,
):
    """
    Train all models, cross-validate, evaluate on holdout.
    Returns (results sorted by cv_score, X_test, y_test, target_encoder).
    """
    cat_cols = [c for c in (cat_cols or []) if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Encode classification target
    if task == "classification":
        # Always label-encode. `y.astype(int)` on a numeric target truncated
        # it: classes 0.2 and 0.7 both became 0, leaving one class, and the
        # one model that still fit reported a meaningless 100% accuracy.
        if target_encoder is None:
            target_encoder = LabelEncoder()
        values = y.astype(str) if (is_text_dtype(y) or pd.api.types.is_bool_dtype(y)) else y
        y = pd.Series(target_encoder.fit_transform(values), index=y.index)

    # Train/test split — stratified for classification, which needs at
    # least two rows of every class or train_test_split raises.
    stratify = (y if task == "classification" and y.nunique() <= MAX_CLASSES
                and y.value_counts().min() >= 2 else None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Classifiers are chosen on log-loss, which charges a model for being
    # confidently wrong. Chosen on F1, a boosted model that said "No" with
    # 99.7% certainty on pure noise beat an honest one. The score shown is
    # accuracy, the same measure as the held-out test score.
    scoring = ({"show": "r2"} if task == "regression"
               else {"show": "accuracy", "pick": "neg_log_loss"})
    # Five folds on small data, three on large: the extra two folds change
    # the estimate little past ~10,000 rows and cost 40% more time.
    folds = 5 if len(X_train) <= 10_000 else 3
    n_classes = int(y.nunique()) if task == "classification" else 0
    results = []

    for name, model in _get_models(task, n_classes, len(X_train)):
        try:
            pipe = _make_pipeline(model, num_cols, cat_cols)

            # Cross-validation on the training set. One job here: the models
            # parallelise internally, and nesting the two oversubscribed the
            # machine.
            cv = cross_validate(pipe, X_train, y_train, cv=folds,
                                scoring=scoring, n_jobs=1, error_score="raise")
            cv_scores = cv["test_show"]
            pick = float(np.mean(cv["test_pick"] if "test_pick" in cv else cv_scores))

            pipe.fit(X_train, y_train)
            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)

            if task == "regression":
                train_s = r2_score(y_train, y_pred_train)
                test_s  = r2_score(y_test,  y_pred_test)
                metrics = _evaluate_regression(y_test, y_pred_test)
                metric_name = "R2"
            else:
                train_s = accuracy_score(y_train, y_pred_train)
                test_s  = accuracy_score(y_test,  y_pred_test)
                try:
                    y_prob = pipe.predict_proba(X_test)
                except Exception:
                    y_prob = None
                metrics = _evaluate_classification(y_test, y_pred_test, y_prob)
                metric_name = "Accuracy"

            gap   = train_s - test_s
            o_lbl = ("None" if gap < 0.05
                     else "Mild" if gap < 0.15
                     else "Severe")

            results.append(ModelResult(
                name=name, task=task,
                cv_score=round(float(np.mean(cv_scores)), 4),
                cv_std=round(float(np.std(cv_scores)), 4),
                train_score=round(float(train_s), 4),
                test_score=round(float(test_s), 4),
                overfit_gap=round(float(gap), 4),
                overfit_label=o_lbl,
                metric_name=metric_name,
                mae=metrics.get("mae"),
                rmse=metrics.get("rmse"),
                f1=metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                model=pipe,
                cv_pick=round(pick, 4),
            ))

        except Exception as e:
            logger.warning(f"Model '{name}' failed to train and was skipped: {e}")
            results.append(ModelResult(
                name=name, task=task,
                cv_score=-999, cv_std=0,
                train_score=0, test_score=0,
                overfit_gap=0, overfit_label="N/A",
                metric_name="N/A",
                model=None,
                cv_pick=-1e9,
            ))

    # Best by the selection score. A model that failed to train is never
    # "best": with every model failed, the sentinel used to be crowned.
    results.sort(key=lambda x: x.cv_pick, reverse=True)
    if results and results[0].model is not None:
        results[0].is_best = True

    return results, X_test, y_test, target_encoder


# ══════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE AND EFFECTS
# ══════════════════════════════════════════════════════════

def _focus_class(task: str, target_encoder, y_train: Optional[pd.Series]):
    """(index, label) of the class whose chance the effects describe: the
    rarer of two — the event (left, churned, defaulted) in practice."""
    if task != "classification" or target_encoder is None:
        return None, None
    if len(target_encoder.classes_) != 2:
        return None, None
    idx = 1
    if y_train is not None and len(y_train):
        counts = pd.Series(y_train).value_counts()
        idx = int(counts.idxmin()) if len(counts) == 2 else 1
    return idx, str(target_encoder.classes_[idx])


def _level(pipe, X: pd.DataFrame, task: str, focus: Optional[int]) -> float:
    """The model's average prediction over X: the target for regression,
    the chance of the focus class for a two-class outcome."""
    if task == "regression":
        return float(np.mean(pipe.predict(X)))
    return float(np.mean(pipe.predict_proba(X)[:, focus]))


def get_feature_importance(
    model_result: ModelResult,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    cat_cols: List[str],
    target_col: str,
    target_encoder=None,
    y_train: Optional[pd.Series] = None,
) -> List[FeatureImportance]:
    """How much the model depends on each column, and which way it pushes.

    Importance is permutation importance on held-out rows: shuffle one
    column and measure how much worse the predictions get. A column the
    model memorised but cannot use on new rows scores about zero. The old
    figures came from each model's own training-time importances, so a
    row-stamp date led the ranking of a model with no skill.

    Direction is measured, not read off a sign: for a number, the model's
    average prediction with the column at its 25th against its 75th
    percentile; for a category, the prediction for each of its values.
    """
    if model_result.model is None or X_test is None or len(X_test) == 0:
        return []
    from sklearn.inspection import permutation_importance
    pipe = model_result.model
    Xs, ys = X_test, y_test
    if len(Xs) > 2000:
        Xs = Xs.sample(2000, random_state=42)
        ys = ys.loc[Xs.index]

    scoring = "r2" if task == "regression" else "accuracy"
    if task == "classification" and hasattr(pipe, "predict_proba"):
        scoring = "neg_log_loss"
    try:
        perm = permutation_importance(pipe, Xs, ys, scoring=scoring, n_repeats=5,
                                      random_state=42, n_jobs=1)
    except Exception:
        logger.info("permutation importance with %s failed; using accuracy", scoring,
                    exc_info=True)
        try:
            perm = permutation_importance(pipe, Xs, ys, scoring=None, n_repeats=5,
                                          random_state=42, n_jobs=1)
        except Exception:
            logger.warning("permutation importance failed", exc_info=True)
            return []

    raw = np.clip(np.asarray(perm.importances_mean, dtype=float), 0, None)
    total = float(raw.sum())
    if total <= 0:
        return []
    share = raw / total

    focus, focus_label = _focus_class(task, target_encoder, y_train)
    effects_ok = task == "regression" or focus is not None
    y_scale = float(np.std(ys)) if task == "regression" else 1.0
    tgt = "'{}'".format(target_col)

    def _amount(d: float) -> str:
        return ("{:.1f} points".format(abs(d) * 100) if task == "classification"
                else fmt(abs(d)))

    out = []
    order = np.argsort(-share)
    for rank, i in enumerate(order, 1):
        col = X_test.columns[i]
        pct = float(share[i]) * 100
        direction, explanation = "mixed", ""
        if effects_ok and pct >= 1:
            try:
                if col in cat_cols:
                    values = X_train[col].value_counts().index[:6]
                    levels = {}
                    for v in values:
                        Xa = Xs.copy()
                        Xa[col] = v
                        levels[str(v)] = _level(pipe, Xa, task, focus)
                    if len(levels) >= 2:
                        hi = max(levels, key=levels.get)
                        lo = min(levels, key=levels.get)
                        show = ((lambda x: "{:.0f}%".format(x * 100)) if task == "classification"
                                else fmt)
                        what = ("the chance of '{}'".format(focus_label)
                                if task == "classification" else tgt)
                        direction = "varies"
                        explanation = ("In the model, {} is highest for '{}' ({}) and "
                                       "lowest for '{}' ({}).".format(
                                           what, hi, show(levels[hi]), lo, show(levels[lo])))
                else:
                    lo_v, hi_v = (float(v) for v in X_train[col].quantile([0.25, 0.75]))
                    if hi_v > lo_v:
                        Xa, Xb = Xs.copy(), Xs.copy()
                        Xa[col], Xb[col] = lo_v, hi_v
                        delta = _level(pipe, Xb, task, focus) - _level(pipe, Xa, task, focus)
                        small = (0.005 if task == "classification"
                                 else 0.01 * (y_scale or 1.0))
                        what = ("a {} chance of '{}'" if task == "classification"
                                else "{} " + tgt)
                        if abs(delta) < small:
                            direction = "flat"
                            explanation = ("Moving '{}' from {} to {} (its middle half) "
                                           "barely changes the prediction.".format(
                                               col, fmt(lo_v), fmt(hi_v)))
                        else:
                            direction = "positive" if delta > 0 else "negative"
                            way = "higher" if delta > 0 else "lower"
                            explanation = ("In the model, '{}' at {} rather than {} goes "
                                           "with {} by {}.".format(
                                               col, fmt(hi_v), fmt(lo_v),
                                               what.format(way, focus_label) if task == "classification"
                                               else what.format(way),
                                               _amount(delta)))
            except Exception:
                logger.debug("effect for %s failed", col, exc_info=True)
        if not explanation:
            explanation = "{:.0f}% of what the model's accuracy depends on.".format(pct)
        out.append(FeatureImportance(feature=str(col), importance=round(float(share[i]), 4),
                                     rank=rank, direction=direction,
                                     explanation=explanation))
    return out[:20]


def describe_feature_inputs(X: pd.DataFrame, cat_cols: List[str],
                            datetime_origins: Dict) -> List[Dict]:
    """What each feature accepts as a what-if input, in the user's terms."""
    out = []
    for col in X.columns:
        s = X[col].dropna()
        if col in cat_cols:
            counts = s.value_counts()
            options = [str(v) for v in counts.index[:200]]
            out.append({"name": col, "kind": "category", "options": options,
                        "default": options[0] if options else None})
        elif col in datetime_origins:
            origin = pd.Timestamp(datetime_origins[col])
            day = lambda d: (origin + pd.Timedelta(days=float(d))).date().isoformat()
            out.append({"name": col, "kind": "date",
                        "min": day(s.min()) if len(s) else None,
                        "max": day(s.max()) if len(s) else None,
                        "default": day(s.median()) if len(s) else None})
        else:
            out.append({"name": col, "kind": "number",
                        "min": float(s.min()) if len(s) else None,
                        "max": float(s.max()) if len(s) else None,
                        "default": float(s.median()) if len(s) else None})
    return out


# ══════════════════════════════════════════════════════════
#  WHAT-IF PREDICTION
# ══════════════════════════════════════════════════════════

def _encode_what_if_inputs(ml_report: MLReport,
                           input_values: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """One feature row in the form the model was trained on.

    Categories go in as their values, dates as days since the earliest
    training date. A field left blank is passed as missing and filled the
    way training filled it — the most common value of a category, the
    median of a number — and is listed back so the caller knows.
    """
    cat_cols = set(getattr(ml_report, "categorical_cols", None) or [])
    origins = getattr(ml_report, "datetime_origins", None) or {}
    options = {f["name"]: set(f.get("options") or [])
               for f in (ml_report.feature_inputs or []) if f.get("kind") == "category"}
    row, assumed = {}, []
    unknown = [k for k in input_values if k not in ml_report.feature_cols]
    if unknown:
        raise ValueError("Not features of this model: {}. Its features are: {}".format(
            ", ".join(map(str, unknown)), ", ".join(ml_report.feature_cols)))
    for col in ml_report.feature_cols:
        if col not in input_values or input_values[col] in (None, ""):
            row[col] = np.nan
            assumed.append(col)
            continue
        v = input_values[col]
        if col in cat_cols:
            known = options.get(col, set())
            if known and str(v) not in known:
                shown = sorted(known)
                raise ValueError("'{}' is not a value '{}' had in training. Known: {}".format(
                    v, col, ", ".join(shown[:20]) + (" …" if len(shown) > 20 else "")))
            row[col] = str(v)
        elif col in origins:
            try:
                when = pd.Timestamp(v)
            except (ValueError, TypeError):
                raise ValueError("'{}' needs a date, got {!r}".format(col, v))
            origin = origins[col]
            if getattr(origin, "tzinfo", None) is not None and when.tzinfo is None:
                when = when.tz_localize(origin.tzinfo)
            row[col] = float((when - origin).days)
        else:
            try:
                row[col] = float(v)
            except (ValueError, TypeError):
                raise ValueError("'{}' needs a number, got {!r}".format(col, v))
    frame = pd.DataFrame([row])[ml_report.feature_cols]
    for col in ml_report.feature_cols:
        frame[col] = frame[col].astype(object) if col in cat_cols else frame[col].astype(float)
    return frame, assumed


def predict_what_if(
    ml_report: MLReport,
    input_values: Dict[str, Any],
) -> Dict:
    """
    Make a single prediction from user-supplied input values.
    Returns prediction + how far to trust it, or {"error": ...}.
    """
    if ml_report.best_model is None or ml_report.best_model.model is None:
        return {"error": "No trained model available."}

    try:
        X_input, assumed = _encode_what_if_inputs(ml_report, input_values)
    except ValueError as e:
        return {"error": str(e)}

    try:
        pipe = ml_report.best_model.model
        pred = pipe.predict(X_input)[0]
        result: Dict[str, Any] = {"prediction": float(pred), "task": ml_report.task,
                                  "target": ml_report.target_col,
                                  "model_score": ml_report.best_model.test_score,
                                  "reliable": not ml_report.no_signal}
        if assumed:
            result["assumed_at_training_median"] = assumed

        if ml_report.task == "classification":
            enc = ml_report.target_encoder
            decode = (lambda c: str(enc.inverse_transform([int(c)])[0])) if enc is not None \
                else (lambda c: str(c))
            result["prediction_label"] = decode(pred)
            try:
                proba = pipe.predict_proba(X_input)[0]
                # Keyed by the real class labels, not the encoder's integers.
                result["probabilities"] = {
                    decode(c): round(float(p), 4)
                    for c, p in zip(pipe.classes_, proba)
                }
                result["confidence"] = round(float(max(proba)) * 100, 1)
            except Exception:
                result["confidence"] = None
        else:
            # The range held-out predictions actually missed by: 8 in 10 of
            # them fell between these errors. This was ±1 RMSE, labelled a
            # confidence interval, which it is not.
            lo_e, hi_e = getattr(ml_report, "residual_band", None) or (None, None)
            if lo_e is not None:
                result["lower"] = round(float(pred) + lo_e, 4)
                result["upper"] = round(float(pred) + hi_e, 4)
                result["confidence_note"] = "8 in 10 held-out predictions fell within this range"

        return result

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
#  INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════

def _beats_chance(report: MLReport) -> bool:
    """Does the classifier do better than chance on the held-out rows?

    For two classes: do rows with the outcome get higher predicted
    probabilities than rows without (a one-sided Mann-Whitney test — the
    significance of the AUC)? Accuracy against the majority class has
    little power on an imbalanced outcome: a real effect on 800 rows went
    unrecognised. For more classes: is accuracy above always answering the
    commonest class by more than chance (binomial test)?
    """
    from scipy.stats import binomtest, mannwhitneyu
    best, y_test, X_test = report.best_model, report.y_test, report.X_test
    try:
        classes = np.unique(y_test)
        if len(classes) == 2 and hasattr(best.model, "predict_proba"):
            p = best.model.predict_proba(X_test)[:, 1]
            pos, neg = p[np.asarray(y_test) == 1], p[np.asarray(y_test) == 0]
            if len(pos) >= 5 and len(neg) >= 5:
                return mannwhitneyu(pos, neg, alternative="greater").pvalue < 0.05
        baseline = max((report.class_balance or {"": 1.0}).values())
        n = len(y_test)
        k = int(round(best.test_score * n))
        return best.test_score > baseline and \
            binomtest(k, n, baseline, alternative="greater").pvalue < 0.05
    except Exception:
        logger.debug("chance test failed", exc_info=True)
        return False


def _no_signal_note(report: MLReport) -> str:
    """Explain, when it applies, that the model has nothing to teach."""
    best = report.best_model
    if best is None:
        return ""
    if report.task == "regression":
        if best.test_score >= 0.05:
            return ""
        # A negative R² is worse than guessing the average, not "barely better".
        how = "worse than" if best.test_score < 0 else "barely better than"
        return ("This model predicts '{}' {} always guessing its "
                "average (R²={:.2f} on held-out rows). Its feature ranking reflects "
                "noise, not influence, and what-if predictions stay near the average "
                "whatever you change.".format(report.target_col, how, best.test_score))
    balance = report.class_balance or {}
    if not balance:
        return ""
    label, baseline = max(balance.items(), key=lambda kv: kv[1])
    # Better than the free baseline by more than chance, on the held-out
    # rows. "One point above" called 53% against 50% on 160 rows a signal.
    n_test = len(report.y_test) if report.y_test is not None else 0
    if n_test and _beats_chance(report):
        return ""
    return ("This model is no better than answering '{}' for every row ({:.1f}% "
            "accuracy against {:.1f}%, a difference within chance on {:,} held-out "
            "rows). Its feature ranking reflects noise, not influence, and what-if "
            "predictions should not be relied on.".format(
                label, best.test_score * 100, baseline * 100, n_test))


def _generate_insights(report: MLReport) -> List[str]:
    insights = []
    best = report.best_model

    if best is None:
        return ["No model trained successfully."]

    # Performance interpretation
    if report.task == "regression":
        r2 = best.test_score
        fit = ("a strong fit" if r2 >= 0.85 else "a good fit" if r2 >= 0.70
               else "a moderate fit" if r2 >= 0.50 else "a weak fit")
        if r2 >= 0.05:
            insights.append(
                "The model explains {:.0f}% of the variation in '{}' on rows it was not "
                "trained on (R²={:.2f}) — {}.".format(r2 * 100, report.target_col, r2, fit))
    else:
        # Judged against always predicting the most common class — the
        # accuracy any model gets for free. This said "Good classifier: 80%
        # accuracy. Better than random by 65.7%" (dividing by the number of
        # *features*) for a model on data that is 80.4% "No": it was worse
        # than never predicting "Yes" at all.
        acc = best.test_score
        balance = report.class_balance or {}
        majority_label, baseline = (max(balance.items(), key=lambda kv: kv[1])
                                    if balance else ("", 0.0))
        lift_pp = (acc - baseline) * 100
        if acc <= baseline + 0.01:
            insights.append(
                "No better than always predicting the most common class: "
                "{:.1f}% accuracy, against {:.1f}% from answering '{}' for "
                "every row.".format(acc * 100, baseline * 100, majority_label))
        elif lift_pp < 5:
            insights.append(
                "Weak classifier: {:.1f}% accuracy, only {:.1f} points above "
                "the {:.1f}% from always answering '{}'.".format(
                    acc * 100, lift_pp, baseline * 100, majority_label))
        else:
            insights.append(
                "Classifier reaches {:.1f}% accuracy on held-out data, {:.1f} "
                "points above the {:.1f}% from always answering '{}'.".format(
                    acc * 100, lift_pp, baseline * 100, majority_label))

    # Overfitting
    if best.overfit_label == "Severe" and not report.no_signal:
        insights.append(
            "It scores {:.2f} on its training rows but {:.2f} on new ones, so expect "
            "its predictions for new records to be noticeably worse than its "
            "training score suggests.".format(best.train_score, best.test_score))
    elif best.overfit_label == "Mild" and not report.no_signal:
        insights.append(
            "It does slightly better on its training rows than on new ones (a gap of "
            "{:.2f}).".format(best.overfit_gap))

    # Top feature — meaningless when the model has no predictive power.
    if report.feature_importance and not report.no_signal:
        top = report.feature_importance[0]
        insights.append(
            "It relies most on '{}' ({:.0f}% of the model).".format(
                top.feature, top.importance * 100))

    # Model comparison — among models that trained; a failed one carries a
    # -999 sentinel score, not a result.
    trained = [m for m in report.models if m.model is not None]
    if len(trained) >= 2 and not report.no_signal:
        best_cv  = trained[0].cv_score
        worst_cv = trained[-1].cv_score
        if best_cv - worst_cv > 0.1:
            insights.append(
                "{} did best of the {} methods tried ({:.2f} against {:.2f} for {}).".format(
                    trained[0].name, len(trained), best_cv, worst_cv, trained[-1].name))
    failed = [m.name for m in report.models if m.model is None]
    if failed:
        insights.append("Could not train: {}.".format(", ".join(failed)))

    # Class imbalance warning
    if report.class_balance:
        max_pct = max(report.class_balance.values())
        if max_pct > 0.80:
            insights.append(
                "Class imbalance detected ({:.0f}% dominant class). "
                "Accuracy metric may be misleading — check F1 score.".format(max_pct*100))

    return insights


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def _leaky_columns(X: pd.DataFrame, y: pd.Series, cat_cols: List[str]) -> List[Tuple[str, str]]:
    """Columns that are (nearly) a copy of the target, with why.

    A number correlated beyond 0.98 with a numeric target, or a category
    whose value settles the outcome on 99% of rows when the outcome itself
    is not that lopsided (a "leaving reason" that is blank for everyone who
    stayed). Either produces a spectacular model that cannot be used,
    because the column is only known once the outcome is.
    """
    out = []
    y_num = pd.to_numeric(y, errors="coerce")
    y_txt = y.astype(str)
    base = float(y_txt.value_counts(normalize=True).iloc[0]) if len(y_txt) else 1.0
    for col in X.columns:
        try:
            if col in cat_cols:
                if y.nunique() > MAX_CLASSES or base > 0.95:
                    continue
                x = X[col].fillna("(blank)")
                if x.nunique() > len(x) / 2:
                    continue
                majority = pd.crosstab(x, y_txt).max(axis=1).sum() / len(x)
                if majority >= 0.99:
                    out.append((col, "its value decides the outcome on {:.0%} of rows"
                                .format(majority)))
            elif y_num.notna().sum() >= 20:
                x = X[col]
                m = x.notna() & y_num.notna()
                if m.sum() < 20 or x[m].nunique() < 2 or y_num[m].nunique() < 2:
                    continue
                r = float(np.corrcoef(x[m], y_num[m])[0, 1])
                if abs(r) > 0.98:
                    out.append((col, "it is almost identical to the target (|r|={:.3f})"
                                .format(abs(r))))
        except Exception:
            logger.debug("leakage check failed for %s", col, exc_info=True)
    return out


def run_ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
    max_rows: int = 50_000,
) -> MLReport:
    """
    Full ML pipeline:
    1. Detect task (regression/classification)
    2. Prepare features
    3. Train + cross-validate multiple models
    4. Feature importance and effects, measured on held-out rows
    5. Generate insights
    Returns MLReport. Raises TargetError for a target that cannot be modelled.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    task, task_reason = detect_task(df[target_col])
    if task == "classification":
        n_classes = int(df[target_col].nunique())
        if n_classes > MAX_CLASSES:
            raise TargetError(
                "'{}' has {} different values — too many to predict as categories. "
                "Pick an outcome with at most {} values, or a numeric measure.".format(
                    target_col, n_classes, MAX_CLASSES))

    class_balance = None
    if task == "classification":
        vc = df[target_col].dropna().astype(str).value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}

    X, y, cat_cols, datetime_origins = prepare_features(df, target_col, selected_features)

    leakage_warnings = []
    for col, why in _leaky_columns(X, y, cat_cols):
        X = X.drop(columns=[col])
        leakage_warnings.append(
            "Left out '{}': {}. It is only known once the outcome is, so a model "
            "using it would look perfect and be useless.".format(col, why))
    cat_cols = [c for c in cat_cols if c in X.columns]
    datetime_origins = {c: o for c, o in datetime_origins.items() if c in X.columns}

    if len(X.columns) == 0:
        report = MLReport(task=task, target_col=target_col, feature_cols=[],
                          n_rows_used=len(df), n_features=0)
        report.warnings.append("No usable feature columns found after preprocessing.")
        return report

    if len(X) < 20:
        report = MLReport(task=task, target_col=target_col, feature_cols=list(X.columns),
                          n_rows_used=len(X), n_features=len(X.columns))
        report.warnings.append("Too few rows ({}) for reliable ML. Need at least 20.".format(len(X)))
        return report

    model_results, X_test, y_test, target_encoder = train_models(
        X, y, task, cat_cols=cat_cols)
    best = next((m for m in model_results if m.is_best), None)
    if best is None:
        leakage_warnings.append(
            "No model could be trained on this target with these features.")

    X_train = X.drop(index=X_test.index)
    y_train = (pd.Series(target_encoder.transform(
                   y.loc[X_train.index].astype(str) if (is_text_dtype(y) or pd.api.types.is_bool_dtype(y))
                   else y.loc[X_train.index]), index=X_train.index)
               if task == "classification" else y.loc[X_train.index])

    y_pred, residual_band = None, None
    if best and best.model is not None:
        try:
            y_pred = best.model.predict(X_test)
            if task == "regression":
                resid = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)
                residual_band = (float(np.quantile(resid, 0.10)),
                                 float(np.quantile(resid, 0.90)))
        except Exception:
            logger.debug("run_ml_pipeline: suppressed exception", exc_info=True)

    report = MLReport(
        task=task,
        target_col=target_col,
        feature_cols=list(X.columns),
        n_rows_used=len(X),
        n_features=len(X.columns),
        class_balance=class_balance,
        models=model_results,
        best_model=best,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        target_encoder=target_encoder,
        datetime_origins=datetime_origins,
        categorical_cols=cat_cols,
        residual_band=residual_band,
        feature_inputs=describe_feature_inputs(X, cat_cols, datetime_origins),
        warnings=[task_reason] + leakage_warnings,
        task_reason=task_reason,
    )

    # Suspiciously perfect scores usually mean leakage we couldn't detect
    if best and best.test_score >= 0.995:
        report.warnings.append(
            "Test score is near-perfect ({:.1%}). Real-world data almost "
            "never behaves like this — check whether a feature contains "
            "information that would not be available before the outcome "
            "(target leakage).".format(best.test_score))

    report.no_signal = _no_signal_note(report)
    if best and best.model is not None:
        report.feature_importance = get_feature_importance(
            best, X_train, X_test, y_test, task, cat_cols, target_col,
            target_encoder=target_encoder, y_train=y_train)
    if report.no_signal:
        for f in report.feature_importance:
            f.direction = "mixed"
            f.explanation = ("{:.0f}% of what the model used — not evidence of an "
                             "effect, since the model does not predict.".format(
                                 f.importance * 100))
    report.insights = _generate_insights(report)
    return report
