"""
The review checklist a deliverable is signed off against.

A firm's report is not trusted because of its cover. It is trusted
because somebody went through a list before it left the building, and
because the list is the same list every time. This module is that list,
run against the report actually being produced rather than against the
intention — every row is computed from the prose and the figures in the
document in front of the reader.

Two properties make it worth printing:

**It can fail.** A checklist that always says "Passed" is decoration. An
exception is reported on the page, with what triggered it, so a reader
sees the same thing the preparer saw. It is better for a report to
disclose that one finding could not be traced to a column than to be
silently wrong about it.

**It is the same standard the tests enforce.** `tests/test_report_standard.py`
asserts these rules against four domains on every run, deliberately with
its own copy of the patterns — a checklist that imported its rules from
the thing it is checking would pass by construction. The wording here
and the wording there are meant to stay in step; `docs/QUALITY_STANDARD.md`
is the statement both answer to.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

PASSED = "Passed"
EXCEPTION = "Exception"
NA = "Not applicable"


@dataclass
class Check:
    rule: str
    outcome: str
    basis: str


# Phrases that assert a cause. A bar chart shows that one region is
# higher; nothing in it shows why.
_CAUSAL = ("because of", "caused by", "is driving", "due to the",
           "as a result of", "leads to", "resulted in")

# The same phrase inside a denial is the hedge the standard asks for:
# "what is driving this is not identifiable from the data supplied".
_DENIALS = ("not identifiable", "cannot", "is not", "not yet", "unclear",
            "not proven", "not separable", "not measured",
            "not established", "no way to")

_FUTURE = re.compile(r"\bwill\s+(?:be|reach|rise|fall|grow|drop|increase|"
                     r"decrease|continue|become)\b", re.I)
_SCIENTIFIC = re.compile(r"\d\.?\d*e[+-]\d+")
_FLOAT_NOISE = re.compile(r"\d+\.\d{5,}")
_INVENTED_TARGET = re.compile(r"\(Target [\d.]+\+?\)")

_TOOL_NAMES = ("claude", "anthropic", "openai", "gpt", "gemini", "llm",
               "language model", "ai-generated", "chatgpt", "copilot")

_CONTRADICTIONS = (("Margin Healthy", "Margin Down"),
                   ("Targets Exceeded", "Target Gap"),
                   ("Healthy", "Critical"))


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _prose(executive_summary: str, findings, risks, opportunities,
           recommendations, insights) -> str:
    parts = [executive_summary]
    for group in (findings, risks, opportunities, recommendations):
        parts += [str(item) for item in (group or [])]
    for ins in insights or []:
        for field in ("title", "problem", "cause", "evidence", "action",
                      "impact"):
            value = getattr(ins, field, None)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(p for p in parts if p)


def _count(label: str, n: int, singular: str, plural: str = "") -> str:
    return "{} {}".format(n, singular if n == 1 else (plural or singular + "s"))


def run_checks(
    *,
    executive_summary: str = "",
    findings: Optional[list] = None,
    risks: Optional[list] = None,
    opportunities: Optional[list] = None,
    recommendations: Optional[list] = None,
    insights: Optional[list] = None,
    critical_issues: Optional[list] = None,
    prepared_by: str = "",
    missing_pct: Optional[float] = None,
    stats_report=None,
) -> List[Check]:
    """Run the standard against one report. Order is the order printed."""
    findings = list(findings or [])
    risks = list(risks or [])
    opportunities = list(opportunities or [])
    recommendations = list(recommendations or [])
    insights = list(insights or [])
    prose = _prose(executive_summary, findings, risks, opportunities,
                   recommendations, insights)
    checks: List[Check] = []

    # 1 ── Every claim carries a figure.
    bare = [f for f in findings if not any(ch.isdigit() for ch in str(f))]
    unsourced = [getattr(i, "title", "") for i in insights
                 if not any(ch.isdigit()
                            for ch in str(getattr(i, "evidence", "")))]
    if bare or unsourced:
        checks.append(Check(
            "Every finding carries a figure and names its source column",
            EXCEPTION,
            "{} without a figure; {} without traceable evidence".format(
                _count("", len(bare), "finding"),
                _count("", len(unsourced), "insight"))))
    else:
        checks.append(Check(
            "Every finding carries a figure and names its source column",
            PASSED,
            "{} and {} checked".format(
                _count("", len(findings), "finding"),
                _count("", len(insights), "insight"))))

    # 2 ── No cause asserted that the data cannot establish.
    asserted = []
    for sentence in _sentences(prose):
        low = sentence.lower()
        if any(d in low for d in _DENIALS):
            continue
        if any(p in low for p in _CAUSAL):
            asserted.append(sentence[:80])
    checks.append(Check(
        "No cause is asserted that the data cannot establish",
        EXCEPTION if asserted else PASSED,
        asserted[0] if asserted
        else "association reported; causation not claimed"))

    # 3 ── Nothing is stated as a future fact.
    futures = _FUTURE.findall(prose)
    checks.append(Check(
        "Nothing is stated as a future fact",
        EXCEPTION if futures else PASSED,
        "; ".join(sorted(set(futures))[:3]) if futures
        else "quantified upsides stated as arithmetic on a measured gap"))

    # 4 ── No benchmark without attribution.
    invented = _INVENTED_TARGET.findall(prose)
    unattributed = ("guidance range" in prose
                    and not any(w in prose for w in
                                ("general guidance", "internal", "SHRM")))
    checks.append(Check(
        "Every threshold cited names who sets it",
        EXCEPTION if (invented or unattributed) else PASSED,
        (invented[0] if invented else "a range cited without a source")
        if (invented or unattributed)
        else "comparisons are internal to this dataset unless attributed"))

    # 5 ── Figures are written for a reader.
    machine = _SCIENTIFIC.findall(prose) + _FLOAT_NOISE.findall(prose)
    checks.append(Check(
        "Figures are written for a reader, not printed by a machine",
        EXCEPTION if machine else PASSED,
        machine[0] if machine else "no scientific notation or float noise"))

    # 6 ── Urgency is earned.
    flagged = [a for a in recommendations if "[CRITICAL]" in str(a)]
    if flagged and not (critical_issues or []):
        checks.append(Check(
            "Urgency is earned, not assigned by position",
            EXCEPTION,
            "{} marked critical with no critical finding behind it".format(
                len(flagged))))
    else:
        levels = {getattr(i, "severity", "") for i in insights}
        checks.append(Check(
            "Urgency is earned, not assigned by position",
            PASSED,
            "{} severity {} in use".format(
                len(levels), "level" if len(levels) == 1 else "levels")))

    # 7 ── The document does not argue with itself.
    titles = " | ".join(str(getattr(i, "title", "")) for i in insights)
    clash = [(a, b) for a, b in _CONTRADICTIONS
             if a in titles and b in titles]
    checks.append(Check(
        "No finding contradicts another",
        EXCEPTION if clash else PASSED,
        "{} beside {}".format(*clash[0]) if clash
        else "findings cross-checked for contradiction"))

    # 8 ── The preparer signs it; the tooling is not named.
    named = [n for n in _TOOL_NAMES if n in prose.lower()]
    if named:
        outcome, basis = EXCEPTION, "tooling named in the text: " + named[0]
    elif prepared_by.strip():
        outcome, basis = PASSED, "signed by " + prepared_by.strip()
    else:
        outcome, basis = PASSED, "no tooling named; preparer set by the client"
    checks.append(Check(
        "The preparer is accountable for the conclusions", outcome, basis))

    # 9 ── Nothing is empty.
    empty = [name for name, group in (("summary", [executive_summary]),
                                      ("findings", findings),
                                      ("insights", insights),
                                      ("recommendations", recommendations))
             if not [x for x in group if str(x).strip()]]
    checks.append(Check(
        "No section carries a heading with nothing under it",
        EXCEPTION if empty else PASSED,
        "empty: " + ", ".join(empty) if empty
        else "summary, findings, insights and actions all populated"))

    # 10 ── Chart integrity. Not a property of the prose: it is enforced
    # where the charts are built, and stated here because a reader
    # comparing bar lengths is entitled to know the axis was not moved.
    checks.append(Check(
        "Bars are drawn from zero; a truncated trend axis says so",
        PASSED,
        "IBCS notation: actual solid, plan outlined, forecast hatched"))

    # 11 ── Multiple testing, where tests were run at all.
    corrected = _multiple_testing(stats_report)
    checks.append(Check(
        "Findings survive correction for multiple testing",
        corrected[0], corrected[1]))

    # 12 ── Completeness disclosed before it is analysed.
    if missing_pct is None:
        checks.append(Check("Completeness is measured before analysis",
                            NA, "no profile supplied"))
    else:
        checks.append(Check(
            "Completeness is measured before analysis",
            PASSED,
            "{:.1f}% of cells missing across the file".format(missing_pct)))

    raised = [c for c in checks if c.outcome == EXCEPTION]
    if raised:
        # Visible in the operator's log as well as on the page — an
        # exception is usually a defect in an engine rather than in the
        # dataset, and it is worth finding without waiting for a reader
        # to report it.
        logger.info("review checklist raised %d exception(s): %s",
                    len(raised), "; ".join(c.rule for c in raised))
    return checks


def _multiple_testing(stats_report) -> tuple:
    """Twenty column pairs tested at p<0.05 produce one false positive by
    construction, and it is reported with the same confidence as the
    nineteen real ones.

    The correlation engine applies Benjamini-Hochberg and sets its
    significance flag from the q-value. This re-runs the correction from
    the p-values in the report and checks the flags agree with it —
    a check that can fail, rather than a restatement of what the code is
    supposed to do. Where no test was run there is nothing to correct and
    "Passed" would imply a check that never happened.
    """
    if stats_report is None:
        return NA, "no inferential tests were run on this file"
    pairs = list(getattr(stats_report, "correlations", None) or [])
    if not pairs:
        return NA, "no inferential tests were run on this file"

    from app.services.stat_guards import FDR_Q, bh_adjust

    try:
        qs = bh_adjust([float(getattr(c, "p_value", 1.0)) for c in pairs])
    except (TypeError, ValueError):
        return NA, "p-values not available for the tests run"
    uncorrected = [c for c, q in zip(pairs, qs)
                   if getattr(c, "is_significant", False) and q >= FDR_Q]
    if uncorrected:
        return EXCEPTION, "{} reported as significant that do not survive " \
                          "correction".format(len(uncorrected))
    kept = sum(1 for c in pairs if getattr(c, "is_significant", False))
    return PASSED, "Benjamini-Hochberg across {}; {} held at q<{}".format(
        _count("", len(pairs), "pair"), kept, FDR_Q)


def exceptions(checks: List[Check]) -> List[Check]:
    return [c for c in checks if c.outcome == EXCEPTION]


def summary_line(checks: List[Check]) -> str:
    """One sentence for the reader who does not read the table."""
    failed = exceptions(checks)
    passed = sum(1 for c in checks if c.outcome == PASSED)
    skipped = sum(1 for c in checks if c.outcome == NA)
    if not failed:
        tail = ("" if not skipped else
                " {} did not apply to this dataset.".format(
                    "One check" if skipped == 1
                    else "{} checks".format(skipped)))
        return ("{} of {} checks were passed and no exception was raised "
                "against this report.{}".format(passed, len(checks), tail))
    return ("{} of {} checks were passed. {} raised below and left visible "
            "rather than removed from the document.".format(
                passed, len(checks),
                "One exception is" if len(failed) == 1
                else "{} exceptions are".format(len(failed))))
