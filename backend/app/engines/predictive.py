"""
core/predictive.py — Model-based predictive drivers for a binary target.

Turns the report from purely descriptive ("attrition is 16%") into predictive
("these factors most predict who leaves, the model separates leavers from
stayers with AUC 0.84, and this profile leaves at 3x the base rate").

No Streamlit. Fully guarded — returns None on any failure so the report can
simply omit the section rather than crash. Uses scikit-learn's RandomForest
(robust to mixed types after light encoding); SHAP is NOT required.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from app.services.dtypes import is_categorical_like

logger = logging.getLogger(__name__)

_BINARY_TARGET_NAMES = ("attrition", "left", "churn", "churned", "exited",
                        "resigned", "terminated", "is_fraud", "default")


@dataclass
class TopCluster:
    description: str      # e.g. "Sales Representative + tenure ≤ 2 yrs"
    n: int                # size of the cluster
    n_events: int         # events (e.g. leavers) inside it
    rate: float           # event rate inside the cluster (%)
    base_rate: float      # overall event rate (%)
    share_of_events: float  # % of ALL events this cluster accounts for


def pick_heatmap_dims(df: pd.DataFrame, target_col: str):
    """Two low-cardinality categorical columns that best separate the target —
    the axes for a 2-way risk heatmap. Returns (dim_a, dim_b) or None."""
    try:
        y = _to_binary(df[target_col])
        if y is None:
            return None
        cats = []
        for c in df.columns:
            if c == target_col:
                continue
            s = df[c]
            if (is_categorical_like(s)
                    or s.nunique() <= 8):
                nun = s.nunique(dropna=True)
                if 2 <= nun <= 9:
                    # variance of group event-rate = how well it separates
                    try:
                        rate = df.assign(_e=y.values).groupby(c, observed=True)["_e"].mean()
                        cats.append((c, float(rate.std()), nun))
                    except Exception:
                        logger.warning("candidate-column event-rate calc failed for '%s'", c, exc_info=True)
        if len(cats) < 2:
            return None
        cats.sort(key=lambda x: x[1], reverse=True)
        # Prefer a small (≤3) dimension as the columns axis for a compact grid.
        a = cats[0][0]
        b = next((c for c, _, n in cats[1:] if n <= 4), cats[1][0])
        return (a, b)
    except Exception:
        logger.warning("pick_heatmap_dims failed", exc_info=True)
        return None


def find_top_cluster(df: pd.DataFrame, target_col: str) -> Optional[TopCluster]:
    """
    The single most concrete, actionable finding: the 2-way segment that
    accounts for the LARGEST share of events while having an elevated rate.
    Produces the reviewer's ideal — "Sales Reps with tenure ≤2 yrs are the
    largest attrition cluster" — instead of a generic template.
    """
    try:
        y = _to_binary(df[target_col])
        if y is None or y.sum() < 10:
            return None
        base = float(y.mean())
        total_events = int(y.sum())
        work = df.copy()
        work["_evt"] = y.values

        # Build candidate segment columns: categoricals (2-12 vals) + binned
        # key numerics (tenure/income/age/hours…) into quartile-ish bands.
        seg_cols = []
        for c in df.columns:
            if c == target_col:
                continue
            s = df[c]
            if is_categorical_like(s) or s.nunique() <= 6:
                if 2 <= s.nunique(dropna=True) <= 12:
                    seg_cols.append(c)
                    work[c] = s.astype(str)
            elif pd.api.types.is_numeric_dtype(s) and s.nunique() > 8 \
                    and any(k in c.lower() for k in
                            ("year", "tenure", "age", "income", "salary", "hour",
                             "rate", "distance", "experience", "month", "charge")):
                try:
                    q = s.quantile([0, .25, .5, .75, 1]).values
                    edges = sorted(set(round(float(e), 2) for e in q))
                    if len(edges) >= 3:
                        labels = []
                        for k in range(len(edges) - 1):
                            lo, hi = edges[k], edges[k + 1]
                            if k == 0:
                                labels.append(f"≤ {hi:g}")
                            elif k == len(edges) - 2:
                                labels.append(f"> {lo:g}")
                            else:
                                labels.append(f"{lo:g}–{hi:g}")
                        band = pd.cut(s, bins=edges, labels=labels,
                                      include_lowest=True).astype(str)
                        work[c + "__band"] = band
                        if 2 <= band.nunique() <= 6:
                            seg_cols.append(c + "__band")
                except Exception:
                    logger.warning("numeric-band candidate calc failed for '%s'", c, exc_info=True)
        if len(seg_cols) < 2:
            return None

        best = None
        # Limit combinatorics — top segments only
        for i in range(len(seg_cols)):
            for j in range(i + 1, len(seg_cols)):
                a, b = seg_cols[i], seg_cols[j]
                try:
                    grp = work.groupby([a, b], observed=True)["_evt"].agg(["mean", "sum", "count"])
                    # A genuine hotspot: rate >= 1.5x base, meaningful size,
                    # and NOT just the majority segment (< 60% of all records).
                    grp = grp[(grp["count"] >= 15)
                              & (grp["mean"] >= base * 1.5)
                              & (grp["count"] <= 0.6 * len(work))]
                    if grp.empty:
                        continue
                    # Score = events contributed x lift — rewards clusters that
                    # are both large AND sharply elevated (a true concentration).
                    grp = grp.assign(_score=grp["sum"] * (grp["mean"] / base))
                    top = grp.sort_values("_score", ascending=False).iloc[0]
                    score = float(top["_score"])
                    if best is None or score > best[4]:
                        va, vb = grp.sort_values("_score", ascending=False).index[0]
                        best = (
                            (a, va, b, vb),
                            int(top["sum"]),
                            int(top["count"]),
                            float(top["mean"]) * 100,
                            score,
                        )
                except Exception:
                    logger.debug("cluster pair failed %s x %s", a, b, exc_info=True)

        if not best:
            return None
        (a, va, b, vb), n_events, n, rate, _ = best

        def _clean(col, val):
            col = col.replace("__band", "").replace("_", " ")
            return f"{col} = {val}"

        desc = f"{_clean(a, va)} AND {_clean(b, vb)}"
        return TopCluster(
            description=desc, n=n, n_events=n_events, rate=rate,
            base_rate=base * 100,
            share_of_events=n_events / max(total_events, 1) * 100,
        )
    except Exception:
        logger.warning("find_top_cluster failed", exc_info=True)
        return None



@dataclass
class DecisionBand:
    """What happens if you act on the top N% of the risk ranking.

    AUC answers "does the model rank correctly". It does not answer the
    question a manager actually asks, which is "we can call 200 customers
    this month — which 200, and how many of them were going to leave
    anyway?" That is precision, recall and lift at a chosen budget, and it
    is the only form in which a model becomes a decision.
    """
    budget_pct: int          # share of the population targeted
    n_targeted: int
    n_events_caught: int
    total_events: int
    precision: float         # % of those targeted who record the event
    recall: float            # % of all events captured
    lift: float              # precision relative to the base rate


def decision_curve(y, proba, budgets=(5, 10, 20, 30)) -> List[DecisionBand]:
    """Precision, recall and lift at each action budget.

    Ranked by predicted probability, so band k contains the k% the model
    considers most at risk.
    """
    bands: List[DecisionBand] = []
    try:
        y_arr = np.asarray(pd.Series(y).astype(int))
        p_arr = np.asarray(proba, dtype=float)
        n = len(y_arr)
        total_events = int(y_arr.sum())
        if n == 0 or total_events == 0:
            return bands
        base = total_events / n
        order = np.argsort(-p_arr)          # highest risk first
        for pct in budgets:
            k = int(round(n * pct / 100.0))
            if k < 1 or k > n:
                continue
            picked = y_arr[order[:k]]
            caught = int(picked.sum())
            precision = caught / k
            bands.append(DecisionBand(
                budget_pct=int(pct), n_targeted=k, n_events_caught=caught,
                total_events=total_events,
                precision=round(precision * 100, 1),
                recall=round(caught / total_events * 100, 1),
                lift=round(precision / base, 2) if base else 0.0))
    except Exception:
        logger.warning("decision curve failed", exc_info=True)
    return bands


def calibration_gap(y, proba, top_pct: int = 20) -> Optional[float]:
    """Percentage points between predicted and observed risk in the top band.

    The high-risk segment is presented as "these records show an X% event
    rate". If the model's own predicted probability for that group is far
    from what actually happened, the number is not safe to quote.
    """
    try:
        y_arr = np.asarray(pd.Series(y).astype(int))
        p_arr = np.asarray(proba, dtype=float)
        k = int(round(len(y_arr) * top_pct / 100.0))
        if k < 10:
            return None
        order = np.argsort(-p_arr)[:k]
        return round(abs(float(p_arr[order].mean()) -
                         float(y_arr[order].mean())) * 100, 1)
    except Exception:
        logger.debug("calibration check failed", exc_info=True)
        return None


@dataclass
class DriverResult:
    target: str
    auc: float
    accuracy: float
    n_rows: int
    n_features: int
    top_drivers: List[Tuple[str, float]] = field(default_factory=list)  # (feature, importance%)
    high_risk_profile: str = ""
    high_risk_rate: float = 0.0
    base_rate: float = 0.0
    high_risk_n: int = 0
    model_name: str = "Random Forest"
    # Whether the model beat the obvious guess. When it did not, the
    # report says so and prints no drivers — importances from a model
    # with no signal describe the noise it was fitted to.
    verdict: Any = None
    leakage: List = field(default_factory=list)
    # Which model was chosen, what it was measured against, whether its
    # scores were calibrated, and where the operating threshold was set.
    model_choice: Any = None
    # What acting on the top N% of the ranking would actually yield.
    decision_bands: List = field(default_factory=list)
    # Precision and recall at the operating threshold, and how far the
    # model's predicted risk is from the observed rate in the top band.
    precision: float = 0.0
    recall: float = 0.0
    calibration_gap: Optional[float] = None


def find_binary_target(df: pd.DataFrame) -> Optional[str]:
    """A column suitable as a prediction target: named like a churn/attrition
    flag AND effectively binary (2 classes, both present, not degenerate)."""
    for col in df.columns:
        cl = col.lower().strip()
        if not any(k in cl for k in _BINARY_TARGET_NAMES):
            continue
        s = df[col].dropna()
        if s.nunique() == 2 and 20 <= len(s):
            vc = s.value_counts(normalize=True)
            if vc.min() >= 0.02:          # minority class at least 2%
                return col
    return None


def _to_binary(s: pd.Series) -> Optional[pd.Series]:
    """Map a 2-class column to 0/1 with 1 = the 'event' (left/yes/true/1)."""
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    if pd.api.types.is_numeric_dtype(s):
        hi = max(vals)
        return (s == hi).astype(int)
    truthy = {"yes", "1", "true", "left", "churned", "y", "t"}
    return s.astype(str).str.lower().str.strip().isin(truthy).astype(int)


# Categorical columns above this many levels are excluded from the model.
# The old cap of 15 silently dropped useful fields; 30 keeps them without
# letting a near-identifier in, and what is excluded is reported.
MAX_CATEGORY_LEVELS = 30


@dataclass
class ModelChoice:
    """Which model was used, how it was tuned, and how far to trust it."""
    name: str
    auc: float
    candidates: List = field(default_factory=list)   # [(name, auc)]
    threshold: float = 0.5
    threshold_basis: str = "default"
    calibrated: bool = False
    calibration_before: Optional[float] = None
    calibration_after: Optional[float] = None
    excluded_high_cardinality: List = field(default_factory=list)


def _candidate_models(n_rows: int, n_features: int):
    """The shortlist: small enough to cross-validate honestly, varied
    enough that one of them usually fits the shape of the data."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("Logistic Regression", Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000,
                                       class_weight="balanced")),
        ])),
    ]
    # Boosting needs enough rows to be worth the variance it adds.
    if n_rows >= 500:
        models.append(("Gradient Boosting", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            random_state=42)))
    return models


def _best_threshold(y, proba) -> Tuple[float, str]:
    """The operating threshold, chosen rather than assumed.

    0.5 is the right cut only when the classes are balanced and the two
    kinds of error cost the same. On the HR sample it gives F1 0.433,
    where 0.22 gives 0.516 — the default was discarding recall the model
    had already earned.
    """
    try:
        from sklearn.metrics import f1_score
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.01):
            f1 = float(f1_score(y, (proba >= t).astype(int), zero_division=0))
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return round(best_t, 2), ("maximises F1 — balances missed cases "
                                  "against false alarms")
    except Exception:
        logger.debug("threshold search failed", exc_info=True)
        return 0.5, "default"


def _select_and_calibrate(X, y, folds: int):
    """Pick the best model by cross-validated AUC, then calibrate it.

    Calibration is kept separate from selection on purpose: ranking
    quality and probability quality are different things. The forest ranks
    well and its raw scores are 19 percentage points out in the top band,
    which makes them unusable as likelihoods even though the ordering is
    sound. Isotonic calibration brings that under 2 points at no
    measurable cost to AUC.

    Returns (proba, fitted_model, ModelChoice), or (None, None, None).
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
    except Exception:
        logger.warning("scikit-learn unavailable", exc_info=True)
        return None, None, None

    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    results, best = [], None
    for name, model in _candidate_models(len(y), X.shape[1]):
        try:
            proba = cross_val_predict(model, X, y, cv=cv,
                                      method="predict_proba", n_jobs=-1)[:, 1]
            auc = float(roc_auc_score(y, proba))
            results.append((name, round(auc, 4)))
            if best is None or auc > best[1]:
                best = (name, auc, model, proba)
        except Exception:
            logger.debug("candidate %s failed", name, exc_info=True)
    if best is None:
        return None, None, None

    name, auc, model, raw_proba = best
    choice = ModelChoice(name=name, auc=round(auc, 4),
                         candidates=sorted(results, key=lambda r: -r[1]))
    proba = raw_proba
    choice.calibration_before = calibration_gap(y, raw_proba)

    # Isotonic needs enough positives per fold to be stable; below that
    # Platt scaling is the safer choice.
    method = "isotonic" if int(np.asarray(y).sum()) >= 150 else "sigmoid"
    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv=3)
        cal_proba = cross_val_predict(calibrated, X, y, cv=cv,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        cal_auc = float(roc_auc_score(y, cal_proba))
        after = calibration_gap(y, cal_proba)
        before = choice.calibration_before
        # Keep it only where it actually improved the probabilities without
        # materially damaging the ranking.
        if (before is not None and after is not None and after < before
                and cal_auc >= auc - 0.02):
            proba = cal_proba
            choice.calibrated = True
            choice.calibration_after = after
            choice.auc = round(cal_auc, 4)
    except Exception:
        logger.debug("calibration failed — using raw scores", exc_info=True)

    if not choice.calibrated:
        choice.calibration_after = choice.calibration_before

    choice.threshold, choice.threshold_basis = _best_threshold(y, proba)

    try:
        model.fit(X, y)
    except Exception:
        logger.warning("final fit failed for %s", name, exc_info=True)
        return proba, None, choice
    return proba, model, choice


def _importances_from(model, columns):
    """Feature importances from whichever model won.

    A forest exposes `feature_importances_`; a logistic regression exposes
    coefficients, whose magnitude on standardised inputs is the comparable
    quantity. Without this, selecting a linear model would silently cost
    the report its drivers.
    """
    try:
        est = model
        if hasattr(est, "named_steps"):
            est = est.named_steps.get("clf", est)
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=columns)
        if hasattr(est, "coef_"):
            return pd.Series(np.abs(np.ravel(est.coef_)), index=columns)
    except Exception:
        logger.debug("importance extraction failed", exc_info=True)
    return None



def compute_drivers(df: pd.DataFrame, target_col: str,
                    max_rows: int = 20000) -> Optional[DriverResult]:
    """Fit a RandomForest to predict the target and return ranked drivers +
    model quality + the highest-risk profile. Returns None if not feasible."""
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
    except Exception:
        logger.warning("scikit-learn unavailable — predictive section skipped")
        return None

    try:
        data = df.dropna(subset=[target_col]).copy()
        if len(data) > max_rows:
            data = data.sample(max_rows, random_state=42)
        y = _to_binary(data[target_col])
        if y is None or y.nunique() < 2 or len(y) < 40:
            return None
        base_rate = float(y.mean()) * 100

        # Feature matrix: numeric as-is, small categoricals one-hot encoded.
        feat = data.drop(columns=[target_col])
        num = feat.select_dtypes(include="number")
        # Drop ID-like / constant numeric columns
        num = num.loc[:, [c for c in num.columns
                          if num[c].nunique() > 1
                          and not any(k in c.lower() for k in ("id", "number", "index"))]]
        cats, excluded_cats = [], []
        for c in feat.select_dtypes(
                include=["object", "string", "category", "bool"]).columns:
            levels = int(feat[c].nunique())
            if 2 <= levels <= MAX_CATEGORY_LEVELS:
                cats.append(c)
            elif levels > MAX_CATEGORY_LEVELS:
                # Named rather than dropped in silence: a field with
                # hundreds of levels may still matter, and the reader
                # should know the model never saw it.
                excluded_cats.append((str(c), levels))
        parts = [num.fillna(num.median(numeric_only=True))]
        col_origin = {c: c for c in num.columns}
        for c in cats:
            dummies = pd.get_dummies(feat[c].astype(str), prefix=c, prefix_sep=" = ")
            for dc in dummies.columns:
                col_origin[dc] = c
            parts.append(dummies)
        X = pd.concat(parts, axis=1)
        if X.shape[1] == 0 or X.shape[0] < 40:
            return None

        # ── Choose a model, calibrate it, choose a threshold ──
        # A single fixed forest was leaving accuracy on the table: on the
        # HR sample a scaled logistic regression scores 0.823 against its
        # 0.803, while on telco the forest wins. Neither is reliably
        # better, which is the argument for choosing per dataset.
        folds = 5 if y.sum() >= 25 else 3
        proba, model, choice = _select_and_calibrate(X, y, folds)
        if choice is None:
            logger.warning("no candidate model fitted for %r", target_col)
            return None
        choice.excluded_high_cardinality = excluded_cats

        auc = float(choice.auc)
        try:
            acc = float(accuracy_score(
                y, (proba >= choice.threshold).astype(int)))
        except Exception:
            logger.debug("accuracy at threshold failed", exc_info=True)
            acc = float("nan")

        # Aggregate one-hot importances back to the original column.
        imp = _importances_from(model, X.columns) if model is not None else None
        if imp is None:
            imp = pd.Series(0.0, index=X.columns)
        agg: dict = {}
        for feat_name, val in imp.items():
            agg[col_origin.get(feat_name, feat_name)] = agg.get(col_origin.get(feat_name, feat_name), 0.0) + float(val)
        total = sum(agg.values()) or 1.0
        ranked = sorted(((c, v / total * 100) for c, v in agg.items()),
                        key=lambda kv: kv[1], reverse=True)[:8]

        # Highest-risk profile: the model's top quintile by predicted
        # probability. proba is positional (row order of X/data), so index
        # positionally with .iloc — not .loc.
        profile, hr_rate, hr_n = "", 0.0, 0
        if proba is not None:
            thr = float(np.quantile(proba, 0.80))
            mask = np.asarray(proba >= thr)
            hr_n = int(mask.sum())
            if hr_n >= 10:
                y_arr = y.to_numpy()
                hr_rate = float(y_arr[mask].mean()) * 100
                hi_rows = data.iloc[mask]
                drivers2 = [c for c, _ in ranked[:2]]
                bits = []
                for c in drivers2:
                    if c in num.columns:
                        med = float(hi_rows[c].median())
                        val = f"{med:,.0f}" if abs(med) >= 100 else f"{med:.2f}"
                        bits.append(f"{c} ≈ {val}")
                    elif c in cats:
                        mode_val = hi_rows[c].astype(str).mode()
                        if len(mode_val):
                            bits.append(f"{c} = '{mode_val.iloc[0]}'")
                profile = "; ".join(bits)

        # ── Turn the ranking into a decision ──────────────────
        bands, precision, recall, calib = [], 0.0, 0.0, None
        if proba is not None:
            bands = decision_curve(y, proba)
            calib = calibration_gap(y, proba)
            try:
                pred = (proba >= choice.threshold).astype(int)
                y_arr = np.asarray(y.astype(int))
                tp = int(((pred == 1) & (y_arr == 1)).sum())
                fp = int(((pred == 1) & (y_arr == 0)).sum())
                fn = int(((pred == 0) & (y_arr == 1)).sum())
                precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) else 0.0
                recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0
            except Exception:
                logger.debug("precision/recall failed", exc_info=True)

        # ── Does this model beat always guessing the majority class? ──
        verdict = None
        leakage = []
        try:
            from app.engines.rigour import assess_classifier, detect_leakage
            if proba is not None:
                verdict = assess_classifier(
                    y, (proba >= 0.5).astype(int), y_proba=proba, auc=auc)
            leakage = detect_leakage(data, target_col)
        except Exception:
            logger.warning("rigour assessment failed for %r", target_col,
                           exc_info=True)

        if verdict is not None and not verdict.usable:
            # No drivers, no high-risk segment, no scenario. The section
            # reports the absence of signal, which is a real finding, and
            # does not dress up a coin flip as a risk model.
            logger.info("predictive: no usable signal for %r (%s)",
                        target_col, verdict.reason)
            return DriverResult(
                target=target_col, auc=auc, accuracy=acc,
                n_rows=len(data), n_features=X.shape[1],
                top_drivers=[], base_rate=base_rate,
                high_risk_profile="", high_risk_rate=0.0, high_risk_n=0,
                verdict=verdict, leakage=leakage, model_choice=choice,
            )

        return DriverResult(
            target=target_col, auc=auc, accuracy=acc,
            n_rows=len(data), n_features=X.shape[1],
            top_drivers=ranked, base_rate=base_rate,
            high_risk_profile=profile, high_risk_rate=hr_rate, high_risk_n=hr_n,
            verdict=verdict, leakage=leakage, decision_bands=bands,
            precision=precision, recall=recall, calibration_gap=calib,
            model_choice=choice, model_name=choice.name,
        )
    except Exception:
        logger.warning("compute_drivers failed", exc_info=True)
        return None
