"""
engines/eda/runner.py — the orchestration.

Decides which columns get which analysis, in what order, and assembles
the report. The only module here that knows about all the others.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


from app.engines.eda.results import EDAReport
from app.engines.eda.univariate import analyze_univariate
from app.engines.eda.bivariate import (analyze_bivariate_numeric,
                                       analyze_group_comparison)
from app.engines.eda.multivariate import analyze_time_series, analyze_vif
from app.engines.eda.findings import _generate_key_findings
from app.engines.plain_language import plain_findings
from app.engines.domains.base import is_id_column
from app.services.dtypes import text_columns


#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, max_rows: int = 50_000) -> EDAReport:
    """
    Full EDA pipeline.
    Returns EDAReport with all analyses.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Identifiers are not measures. EmployeeNumber was being profiled,
    # correlated, VIF-tested and compared across departments — a full
    # statistical treatment of a row number, printed beside the findings
    # about salary.
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in all_numeric if not is_id_column(c, df[c])]
    id_cols  = [c for c in all_numeric if c not in num_cols]
    cat_cols  = text_columns(df)
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()

    report = EDAReport(
        n_rows=len(df), n_cols=len(df.columns),
        numeric_cols=num_cols, categorical_cols=cat_cols,
        datetime_cols=dt_cols,
    )
    report.identifier_cols = id_cols

    # 1. Univariate — every column that is a measure
    analysable = [c for c in df.columns if c not in id_cols]
    for col in analysable[:30]:
        try:
            report.univariate[col] = analyze_univariate(df[col])
        except Exception:
            logger.debug("run_eda: suppressed exception", exc_info=True)
            continue

    # 2. Correlations — numeric pairs
    normality = {
        col: report.univariate[col].is_normal or False
        for col in num_cols if col in report.univariate
    }
    for i in range(len(num_cols)):
        for j in range(i+1, min(len(num_cols), i+8)):
            col_a, col_b = num_cols[i], num_cols[j]
            try:
                res = analyze_bivariate_numeric(
                    df, col_a, col_b,
                    normality.get(col_a, False),
                    normality.get(col_b, False),
                )
                report.correlations.append(res)
            except Exception:
                logger.debug("run_eda: suppressed exception", exc_info=True)
                continue

    # Benjamini-Hochberg across the whole family of pairwise tests:
    # is_significant is re-decided on adjusted p, so testing many pairs
    # doesn't manufacture "significant" findings.
    if report.correlations:
        from app.services.stat_guards import FDR_Q, bh_adjust
        qvals = bh_adjust([c.p_value for c in report.correlations])
        for c, q in zip(report.correlations, qvals):
            c.is_significant = bool(q < FDR_Q)
            c.p_value = float(q)   # store adjusted p (q-value)

    report.correlations.sort(
        key=lambda x: abs(x.statistic), reverse=True
    )

    # 3. Group comparisons — top numeric vs top categorical
    useful_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 15]
    for cat in useful_cats[:3]:
        for num in num_cols[:3]:
            try:
                res = analyze_group_comparison(
                    df, num, cat, normality.get(num, False)
                )
                report.group_comparisons.append(res)
            except Exception:
                logger.debug("run_eda: suppressed exception", exc_info=True)
                continue

    # 4. Multicollinearity
    if len(num_cols) >= 2:
        try:
            report.multicollinearity = analyze_vif(df)
        except Exception:
            logger.debug("run_eda: suppressed exception", exc_info=True)

    # 5. Time series
    if dt_cols and num_cols:
        for dt_col in dt_cols[:1]:
            for num_col in num_cols[:2]:
                try:
                    res = analyze_time_series(df, dt_col, num_col)
                    report.time_series.append(res)
                except Exception:
                    logger.debug("run_eda: suppressed exception", exc_info=True)
                    continue

    # 6. Depth layer
    try:
        from app.engines.eda_depth import (
            describe_imbalance, find_interactions, find_rare_categories,
            key_estimates,
        )
        report.estimates = key_estimates(df)
        report.interactions = find_interactions(df)
        report.rare_categories = find_rare_categories(df)
        report.imbalance_notes = describe_imbalance(df)
    except Exception:
        logger.warning("EDA depth layer failed", exc_info=True)

    # 7. Key findings
    report.key_findings = _generate_key_findings(report)
    # Both readings ship together. The technical list is what an analyst
    # checks the work against; the plain list is what everyone else can
    # act on. Neither is a summary of the other.
    report.plain_findings = plain_findings(report)

    return report

