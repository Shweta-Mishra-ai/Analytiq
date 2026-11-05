"""
domains/pipeline.py — a CRM opportunity export: where the pipeline is,
what is stuck or slipping, how often it converts, and whether it covers
the number.

Method:

  * "Today" is the latest date in the file, not the calendar: an export
    analysed a month after it was taken would otherwise call every open
    deal overdue. The date used is stated.
  * Open and closed deals are told apart by the stage (or status) text:
    "Closed Won", "Won", "Closed Lost", "Lost". Win rate is won over won
    plus lost; open deals are not losses.
  * Risk flags follow a standard pipeline review: a close date already in
    the past, no activity for 14 days, a deal in its stage more than twice
    that stage's median age, and no next step. Each flag needs its own
    column; without it the flag is not raised and the report says so.
  * Weighted pipeline uses the file's own probability column; a
    probability is never invented from a stage name.
  * Coverage is open pipeline over the remaining gap to quota, against the
    common three-times rule of thumb, only when a quota is in the file.
  * Quota attainment is total won over total quota, per rep and overall.
    Averaging each row's attainment (or dividing mean revenue by mean
    target) weights a rep with one tiny deal like the whole team.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.engines.domains.base import build_insight, is_id_column

logger = logging.getLogger(__name__)

STALE_DAYS = 14
_WON = re.compile(r"\b(?:closed[\s_-]*)?won\b|\bwin\b|\bbooked\b", re.I)
_LOST = re.compile(r"\b(?:closed[\s_-]*)?lost\b|\blose\b|\bdisqualified\b|\bdead\b", re.I)


def _words(name) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _find(df, keys, kind, exclude=()):
    from app.services.dtypes import text_columns
    pool = (df.select_dtypes(include="number").columns if kind == "num"
            else df.select_dtypes(include="datetime").columns if kind == "date"
            else text_columns(df))
    for c in pool:
        ws = _words(c)
        joined = " ".join(ws)
        if any(k in joined for k in keys) and not any(x in joined for x in exclude):
            if kind == "num" and (pd.api.types.is_bool_dtype(df[c]) or is_id_column(c, df[c])):
                continue
            return c
    return None


@dataclass
class PipelineColumns:
    stage: Optional[str] = None
    amount: Optional[str] = None
    close_date: Optional[str] = None
    created: Optional[str] = None
    last_activity: Optional[str] = None
    next_step: Optional[str] = None
    probability: Optional[str] = None
    owner: Optional[str] = None
    quota: Optional[str] = None
    forecast: Optional[str] = None

    @property
    def usable(self) -> bool:
        return bool(self.stage and self.amount)


def detect_columns(df: pd.DataFrame) -> PipelineColumns:
    c = PipelineColumns(
        stage=_find(df, ("stage", "status", "deal phase", "pipeline step"), "text",
                    exclude=("forecast",)),
        amount=_find(df, ("amount", "deal value", "deal size", "value", "revenue", "acv", "arr",
                          "tcv", "booking"), "num",
                     exclude=("probability", "weighted", "quota", "target", "pct", "percent")),
        close_date=_find(df, ("close", "expected", "closing"), "date"),
        created=_find(df, ("created", "open date", "opened", "start"), "date"),
        last_activity=_find(df, ("activity", "last contact", "last touch", "modified", "updated"),
                            "date"),
        next_step=_find(df, ("next step", "next action"), "text"),
        probability=_find(df, ("probability", "win chance", "likelihood", "confidence"), "num"),
        owner=_find(df, ("owner", "rep", "salesperson", "account manager", "seller", "ae"),
                    "text"),
        quota=_find(df, ("quota", "target", "goal"), "num"),
        forecast=_find(df, ("forecast category", "forecast", "commit"), "text"),
    )
    if c.stage:
        stages = df[c.stage].dropna().astype(str)
        # A "stage" column with no recognisable pipeline vocabulary and too
        # many values is something else (a free-text status).
        if stages.nunique() > 25:
            c.stage = None
    return c


def _status(stage: pd.Series) -> pd.Series:
    s = stage.astype("string").fillna("")
    out = pd.Series("open", index=s.index)
    out[s.str.contains(_LOST)] = "lost"
    out[s.str.contains(_WON) & ~s.str.contains(r"\bnot\b", case=False)] = "won"
    return out


def _money(x: float) -> str:
    from app.engines.chart_exporter import fmt
    return fmt(x)


def run_pipeline(df: pd.DataFrame, insights: List, findings: List, risks: List,
                 opps: List) -> Optional[dict]:
    c = detect_columns(df)
    if not c.usable:
        return None
    try:
        return _analyse(df, c, insights, findings, risks, opps)
    except Exception:
        logger.warning("pipeline analysis failed", exc_info=True)
        return None


def _analyse(df, c, insights, findings, risks, opps) -> Optional[dict]:
    data = df.copy()
    data["_amt"] = pd.to_numeric(data[c.amount], errors="coerce").fillna(0).clip(lower=0)
    data["_status"] = _status(data[c.stage])
    open_ = data[data["_status"] == "open"]
    won, lost = data[data["_status"] == "won"], data[data["_status"] == "lost"]
    if len(data) < 10 or (open_.empty and won.empty):
        return None

    # "Today" from dates that have happened (created, last activity). A
    # close date on an open deal is a forecast, often months ahead: taking
    # its maximum put "today" in the future, aged every deal by months and
    # flagged the whole pipeline. Close dates count only on closed deals.
    seen = [pd.to_datetime(data[x], errors="coerce") for x in (c.created, c.last_activity) if x]
    seen = [x for x in seen if x.notna().any()]
    if c.close_date and not seen:
        closed_mask = data["_status"] != "open"
        seen.append(pd.to_datetime(data.loc[closed_mask, c.close_date], errors="coerce"))
    asof = max((s.max() for s in seen if s.notna().any()), default=pd.NaT)
    asof_txt = asof.strftime("%d %b %Y") if pd.notna(asof) else ""
    result = {"columns": c, "asof": asof, "open_count": len(open_),
              "open_value": float(open_["_amt"].sum())}

    # ── stage rollup ─────────────────────────────────────
    by_stage = open_.groupby(c.stage, observed=True)["_amt"].agg(["count", "sum"])
    if c.created and pd.notna(asof):
        age = (asof - pd.to_datetime(open_[c.created], errors="coerce")).dt.days
        by_stage["median_age"] = age.groupby(open_[c.stage]).median()
    by_stage = by_stage.sort_values("sum", ascending=False)
    result["by_stage"] = by_stage
    if len(open_):
        parts = ["{}: {} deals worth {}".format(k, int(r["count"]), _money(r["sum"]))
                 + (", median {:.0f} days old".format(r["median_age"])
                    if "median_age" in r and np.isfinite(r["median_age"]) else "")
                 for k, r in by_stage.head(8).iterrows()]
        findings.append("Open pipeline{}: {} deals worth {}. By stage: {}.".format(
            " as of " + asof_txt if asof_txt else "", len(open_),
            _money(open_["_amt"].sum()), " · ".join(parts)))

    # ── weighted pipeline ────────────────────────────────
    weighted = None
    if c.probability and len(open_):
        p = pd.to_numeric(open_[c.probability], errors="coerce")
        if p.max() > 1.5:
            p = p / 100
        p = p.clip(0, 1)
        weighted = float((open_["_amt"] * p.fillna(0)).sum())
        result["weighted"] = weighted
        findings.append("Weighted by the file's own probabilities, the open pipeline is "
                        "worth {} of its {}.".format(_money(weighted),
                                                     _money(open_["_amt"].sum())))

    # ── win rate and cycle ───────────────────────────────
    closed = len(won) + len(lost)
    if closed >= 10:
        rate = len(won) / closed
        val_rate = won["_amt"].sum() / max(won["_amt"].sum() + lost["_amt"].sum(), 1e-9)
        line = ("Win rate {:.0f}% of closed deals ({} won, {} lost); {:.0f}% by value."
                .format(rate * 100, len(won), len(lost), val_rate * 100))
        if c.created and c.close_date:
            cyc = (pd.to_datetime(won[c.close_date], errors="coerce")
                   - pd.to_datetime(won[c.created], errors="coerce")).dt.days
            cyc = cyc[cyc >= 0]
            if len(cyc) >= 5:
                line += " Won deals took a median {:.0f} days from creation to close.".format(
                    cyc.median())
                result["median_cycle"] = float(cyc.median())
        findings.append(line)
        result["win_rate"] = rate
        if val_rate < rate - 0.1:
            risks.append("Large deals are lost more often than small ones: {:.0f}% of closed "
                         "deals are won but only {:.0f}% of closed value.".format(
                             rate * 100, val_rate * 100))

    # ── risk flags on open deals ─────────────────────────
    flags: Dict[str, pd.Series] = {}
    missing: List[str] = []
    if pd.notna(asof):
        if c.close_date:
            cd = pd.to_datetime(open_[c.close_date], errors="coerce")
            flags["close date already passed"] = cd < asof.normalize()
        else:
            missing.append("close date")
        if c.last_activity:
            la = pd.to_datetime(open_[c.last_activity], errors="coerce")
            flags["no activity for {}+ days".format(STALE_DAYS)] = (asof - la).dt.days >= STALE_DAYS
        else:
            missing.append("last activity")
        if c.created:
            age = (asof - pd.to_datetime(open_[c.created], errors="coerce")).dt.days
            med = age.groupby(open_[c.stage]).transform("median")
            flags["in its stage over twice the median age"] = (age > 2 * med) & (med > 0)
    if c.next_step:
        ns = open_[c.next_step].astype("string").str.strip()
        flags["no next step"] = ns.isna() | (ns == "")
    else:
        missing.append("next step")

    if flags and len(open_):
        any_flag = pd.concat(flags.values(), axis=1).fillna(False).any(axis=1)
        at_risk = open_[any_flag]
        result["at_risk_count"] = int(any_flag.sum())
        result["at_risk_value"] = float(at_risk["_amt"].sum())
        counts = " · ".join("{}: {} deals, {}".format(k, int(v.sum()),
                                                     _money(open_.loc[v.fillna(False), "_amt"].sum()))
                           for k, v in flags.items() if v.sum())
        share = at_risk["_amt"].sum() / max(open_["_amt"].sum(), 1e-9)
        if len(at_risk):
            top = at_risk.sort_values("_amt", ascending=False).head(3)
            name_col = next((x for x in df.columns if _words(x)[:1] and _words(x)[0] in
                             ("opportunity", "deal", "account", "name", "company")), None)
            names = ", ".join("{} ({})".format(r[name_col] if name_col else "row {}".format(i + 2),
                                               _money(r["_amt"])) for i, r in top.iterrows())
            insights.append(build_insight(
                title="{} open deals worth {} ({:.0f}% of open pipeline) carry a risk flag"
                      .format(len(at_risk), _money(at_risk["_amt"].sum()), share * 100),
                problem="Flags: {}.".format(counts),
                cause="A past close date or a quiet deal is the commonest sign that the "
                      "stage no longer reflects reality; the file shows the symptom, not "
                      "the reason.",
                evidence="As of {} (the latest date in the file). Largest flagged: {}.{}".format(
                    asof_txt, names, " Not checked for lack of a column: {}.".format(
                        " · ".join(missing)) if missing else ""),
                action="1. Review the largest flagged deals with their owners this week  "
                       "2. Move close dates that have passed, or close the deal as lost  "
                       "3. Give every open deal a dated next step",
                impact="Flagged value is the part of the forecast most likely to slip.",
                severity="high" if share >= 0.3 else "warning", category="pipeline_risk"))
            risks.append("{:.0f}% of open pipeline value ({}) has a risk flag.".format(
                share * 100, _money(at_risk["_amt"].sum())))

    # ── forecast buckets ─────────────────────────────────
    if c.forecast and len(open_):
        b = open_.groupby(open_[c.forecast].astype("string").fillna("(blank)"),
                          observed=True)["_amt"].agg(["count", "sum"]).sort_values(
                              "sum", ascending=False)
        findings.append("Forecast categories of open deals: {}. Closed won so far: {} "
                        "deals, {}.".format(
                            " · ".join("{}: {} deals, {}".format(k, int(r["count"]), _money(r["sum"]))
                                      for k, r in b.iterrows()),
                            len(won), _money(won["_amt"].sum())))

    # ── quota: attainment and coverage ───────────────────
    if c.quota:
        _quota(data, won, open_, c, weighted, insights, findings, risks, opps, result)
    return result


def _quota(data, won, open_, c, weighted, insights, findings, risks, opps, result):
    q = pd.to_numeric(data[c.quota], errors="coerce")
    if c.owner:
        # Quota repeats on every row of a rep; count it once per rep.
        per_rep_q = q.groupby(data[c.owner]).max()
        quota_total = float(per_rep_q.sum())
        won_by = won.groupby(c.owner)["_amt"].sum().reindex(per_rep_q.index).fillna(0)
        att = (won_by / per_rep_q).replace([np.inf, -np.inf], np.nan).dropna()
        if len(att) >= 3:
            at_q = int((att >= 1).sum())
            result["reps_at_quota"] = at_q
            findings.append(
                "{} of {} reps have reached quota on closed-won value; team attainment is "
                "{:.0f}% (total won over total quota, not an average of rep "
                "percentages).".format(at_q, len(att), won_by.sum() / quota_total * 100))
            lo, hi = att.idxmin(), att.idxmax()
            if att[hi] - att[lo] >= 0.5:
                insights.append(build_insight(
                    title="Rep attainment ranges from {:.0f}% ({}) to {:.0f}% ({})".format(
                        att[lo] * 100, lo, att[hi] * 100, hi),
                    problem="{} of {} reps are at or above quota.".format(at_q, len(att)),
                    cause="Attainment reflects territory and pipeline as much as skill; "
                          "compare each rep's open pipeline before drawing conclusions.",
                    evidence="Closed-won value over quota, per rep: {}.".format(" · ".join(
                        "{}: {:.0f}%".format(k, v * 100) for k, v in
                        att.sort_values(ascending=False).head(10).items())),
                    action="1. Check pipeline coverage per rep, not only attainment  "
                           "2. Pair the lowest-attaining reps with deal reviews on their "
                           "largest open deals",
                    impact="", severity="warning" if at_q / len(att) < 0.5 else "info",
                    category="pipeline_attainment"))
    else:
        quota_total = float(q.max())
    if quota_total > 0:
        gap = quota_total - float(won["_amt"].sum())
        basis = weighted if weighted is not None else float(open_["_amt"].sum())
        if gap > 0 and basis > 0:
            cov = basis / gap
            result["coverage"] = cov
            line = ("{} pipeline covers the remaining gap to quota ({}) {:.1f} times"
                    .format("Weighted" if weighted is not None else "Open", _money(gap), cov))
            findings.append(line + (" — under the usual 3x." if cov < 3 else "."))
            if cov < 3 and weighted is None:
                risks.append("Open pipeline is {:.1f}x the gap to quota, below the 3x that "
                             "typical win rates need.".format(cov))
            elif cov < 1:
                risks.append("Even weighted pipeline won in full would not close the gap to "
                             "quota ({:.1f}x).".format(cov))
        elif gap <= 0:
            opps.append("Quota is already met on closed-won value ({} won against {}).".format(
                _money(won["_amt"].sum()), _money(quota_total)))
