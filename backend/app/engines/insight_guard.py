"""
engines/insight_guard.py — the last check before a finding is printed.

Everything upstream tries to be careful: significance tests are corrected
for multiplicity, effect floors are applied, causes are hedged. This
module assumes all of that can still fail, and checks the finished text
of each finding against what a reader can verify.

It removes three things:

  1. **Findings with no number.** "Revenue performance is concerning" is
     an opinion. A reader cannot check it, cannot act on it, and cannot
     tell whether the analysis found anything at all.
  2. **Predictions stated as fact.** "Attrition will reach 30% next
     quarter" is a claim about a future this data cannot observe. The
     finding survives only if it is framed as a projection with its
     assumption stated — a straight-line extrapolation is a legitimate
     thing to show and an illegitimate thing to assert.
  3. **Causal claims the data cannot support.** "Low pay is causing
     attrition" from a dataset that measures both is a correlation
     wearing a verb it has not earned.

It deliberately does not try to judge whether a finding is *interesting*.
Removing a boring but supported finding is a worse error than printing
one, because the reader can skip a dull paragraph and cannot detect an
unfounded one.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

_HAS_NUMBER = re.compile(r"\d")

# Verbs that assert one thing produced another. Correlation analysis
# cannot support any of them on observational data.
_CAUSAL_VERBS = (
    "causes", "caused by", "causing", "drives", "driven by", "driving",
    "leads to", "results in", "resulting in", "produces", "because of",
    "due to", "responsible for", "is why", "explains why",
)
# ... unless the sentence also says what has not been established.
_HEDGES = (
    "not established", "not identifiable", "cannot", "could not",
    "not proven", "not confirmed", "would narrow", "to confirm",
    "confirm which", "requires", "verify", "association", "correlat",
    "consistent with", "may ", "suggests", "appears", "not evidence",
    "not, on its own", "does not establish", "hypothesis", "worth checking",
    "not in this data", "records orders only", "not tested",
)
# "will <verb>" asserts an outcome the data cannot observe. Matched as a
# pattern rather than a list of verbs: an enumeration missed "will
# collapse", and the next dataset would produce a verb nobody listed.
_WILL_VERB = re.compile(r"\bwill\s+(\w+)", re.I)
_OTHER_FUTURE_ASSERTIONS = (
    "is going to", "guarantees", "guaranteed", "ensures", "certain to",
)
# "will require a review" is advice about what to do next, not a forecast
# of an outcome, and filtering it would strip the action out of findings
# that are otherwise sound.
_NON_PREDICTIVE_WILL = {
    "require", "need", "help", "show", "depend", "vary", "differ",
    "involve", "mean", "take", "allow", "let", "give", "tell", "remain",
    "continue",   # "will continue to be reviewed" — an intent, not a claim
}
_PROJECTION_MARKERS = (
    "projection", "projected", "if the same", "at the current rate",
    "straight-line", "extrapolat", "assuming", "holds", "would ",
    "estimate", "indicative", "scenario",
)


@dataclass
class GuardResult:
    kept: List
    dropped: List      # (insight, reason)

    @property
    def drop_reasons(self) -> List[str]:
        return [reason for _ins, reason in self.dropped]


def _text_of(ins, *fields) -> str:
    parts = []
    for f in fields:
        v = ins.get(f, "") if isinstance(ins, dict) else getattr(ins, f, "")
        parts.append(str(v or ""))
    return " ".join(parts)


def _is_quantified(ins) -> bool:
    """A finding a reader can check carries a figure somewhere."""
    return bool(_HAS_NUMBER.search(
        _text_of(ins, "title", "problem", "evidence")))


def _asserts_a_future(ins) -> bool:
    blob = _text_of(ins, "title", "problem", "impact", "action").lower()

    predicted = any(
        verb.lower() not in _NON_PREDICTIVE_WILL
        for verb in _WILL_VERB.findall(blob))
    if not predicted and not any(w in blob for w in _OTHER_FUTURE_ASSERTIONS):
        return False
    # A projection that names itself as one is fine — that is how a
    # forecast is properly stated.
    return not any(m in blob for m in _PROJECTION_MARKERS)


def _asserts_causation(ins) -> bool:
    cause = _text_of(ins, "cause", "problem", "title").lower()
    if not any(v in cause for v in _CAUSAL_VERBS):
        return False
    return not any(h in cause for h in _HEDGES)


def guard_insights(insights: List) -> GuardResult:
    """Drop findings that a reader could not check or should not trust.

    Returns both halves. The dropped list is not discarded by callers —
    knowing that three findings were withheld, and why, is itself
    information about the dataset.
    """
    kept, dropped = [], []
    for ins in insights or []:
        try:
            if not _is_quantified(ins):
                dropped.append((ins, "no figure a reader could check"))
                continue
            if _asserts_a_future(ins):
                dropped.append(
                    (ins, "stated a future outcome as fact rather than as a "
                          "projection with its assumption"))
                continue
            if _asserts_causation(ins):
                dropped.append(
                    (ins, "asserted cause without saying what the data does "
                          "not establish"))
                continue
            kept.append(ins)
        except Exception:
            logger.warning("insight guard failed on an item", exc_info=True)
            kept.append(ins)
    if dropped:
        logger.info("insight guard withheld %d finding(s)", len(dropped))
    return GuardResult(kept=kept, dropped=dropped)


def withheld_note(result: GuardResult) -> str:
    """One sentence for the report about what was held back, or "".

    Printed rather than hidden: a reader who is told that two findings
    were withheld for lack of evidence trusts the ones that remain more,
    not less.
    """
    if not result.dropped:
        return ""
    reasons = sorted(set(result.drop_reasons))
    return (
        "{} candidate finding(s) were produced by the analysis and withheld "
        "from this report because they did not meet the evidence standard "
        "applied throughout it ({}).".format(
            len(result.dropped), "; ".join(reasons)))
