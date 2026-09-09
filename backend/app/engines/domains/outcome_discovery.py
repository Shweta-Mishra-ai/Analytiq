"""
domains/outcome_discovery.py — finding the outcome nobody declared.

Every domain engine in this app knows the name of the thing it cares
about: the HR engine looks for `Attrition`, the SaaS engine for `churned`,
the sales engine for `won`. `outcome_rates` then answers the question a
reader actually has — *who does this happen to, and how much more often
than everyone else* — with a Fisher test behind it.

None of that ran for an ordinary upload. A file with no domain signature
fell to the general engine, which reports the shape of the data, one
group difference on a median, and a correlation. On an operations export
whose `rework` flag sits at 31% for one site against 11% everywhere else
— a real, significant, actionable fact, the single most useful line in
the file — it said nothing at all, because no keyword list had heard of
"rework".

Most uploads are that file. So this module finds binary outcome columns
by their *shape* rather than their name, works out which end of each one
is the problem, and hands them to the same analysis the named domains
get.

Naming a column still helps and is still used: a recognised word sets the
direction, so a win rate is reported by its worst group and a defect rate
by its worst group, which are opposite ends of the table. An unrecognised
flag is analysed just the same and described without a verdict — the
concentration is a fact; calling it good or bad is not this module's to
decide.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from app.engines.domains._common import binary_mask

logger = logging.getLogger(__name__)

# A flag that almost never fires, or almost always does, has no groups
# left to compare — 40 events across six segments is a story about
# sampling, not about the business.
MIN_RATE_PCT = 1.0
MAX_RATE_PCT = 99.0
MIN_EVENTS = 30

# How many outcome columns are worth reporting on. A file can carry a
# dozen flags; a report that works through all of them is a data
# dictionary, not an analysis.
MAX_OUTCOMES = 2

# A column has to be usable as a grouping dimension for the analysis to
# have anywhere to go.
MIN_ROWS = 200

# Words that say the flag firing is the bad thing.
_BAD = (
    "churn", "attrit", "turnover", "resign", "quit", "leave", "left",
    "return", "refund", "chargeback", "dispute", "fraud", "default",
    "delinquen", "cancel", "cancelled", "canceled", "fail", "failure",
    "error", "defect", "rework", "scrap", "reject", "complaint",
    "escalat", "late", "delay", "overdue", "breach", "violation",
    "incident", "accident", "injur", "readmit", "readmission",
    "relapse", "absent", "noshow", "no_show", "missed", "abandon",
    "unsubscribe", "bounce", "opt_out", "optout", "lost", "loss",
    "stockout", "backorder", "damage", "spoil", "downtime", "outage",
    "fault", "risk", "flagged", "suspicious", "overrun",
)

# Words that say the flag firing is the good thing — the report then
# leads with the group at the BOTTOM, because a 42% win rate is not a
# critical incident.
_GOOD = (
    "won", "win", "convert", "conversion", "success", "approved",
    "accept", "retain", "renew", "resubscrib", "subscrib", "purchas",
    "order", "complete", "deliver", "fulfil", "fulfill", "resolved",
    "satisfied", "promot", "graduat", "pass", "qualified", "engaged",
    "active", "repeat", "upsell", "cross_sell", "adopted", "onboard",
    "ontime", "on_time", "attend", "recovered", "cured", "hired",
    "paid", "responded", "clicked", "opened", "signup", "sign_up", "registered",
)

# Words that invert whatever follows them.
_NEGATIONS = {"not", "no", "non", "never", "without", "un", "failed",
              "missing"}

# The same, run together with the word: `undelivered`, `noncompliant`.
_NEGATION_PREFIXES = ("un", "non", "not")

# Prefixes and suffixes that carry no meaning in a sentence.
_STRIP = ("is", "has", "was", "did", "flag", "ind", "indicator", "status",
          "yn", "bool", "boolean")


def _words(name: str) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


def _noun_for(column: str) -> str:
    """A phrase that reads in "Highest ___: site 'Delta' at 31%".

    `is_rework` becomes "rework"; `OnTimeDelivery` becomes "on time
    delivery". Nothing clever — the column name is the user's own word
    for the thing, and a synonym would only make the report harder to
    trace back to the data.
    """
    parts = [w for w in _words(column) if w.lower() not in _STRIP]
    if not parts:
        parts = _words(column) or [str(column)]
    return " ".join(p.lower() for p in parts)[:40]


def _direction(column: str) -> Optional[bool]:
    """True when the flag firing is good, False when bad, None when the
    name does not say.

    Longest match wins so `no_show` is not read as `show`, and the bad
    list is consulted first because a negation usually reads as a
    problem word ("not_delivered" carries "deliver").
    """
    words = [w.lower() for w in _words(column)]
    if not words:
        return None
    flat = "".join(words)

    bad, good = _keywords_in(words)
    if not bad and not good:
        # `undelivered` and `noncompliant` carry no whole word from
        # either list until the prefix comes off.
        for prefix in _NEGATION_PREFIXES:
            if not flat.startswith(prefix) or len(flat) <= len(prefix) + 3:
                continue
            bad, good = _keywords_in([flat[len(prefix):]])
            if bad or good:
                # The prefix is the negation, so the polarity inverts.
                return not (len(good) > len(bad))
        return None

    verdict = len(good) > len(bad)
    matched = good if verdict else bad
    if _is_negated(words, matched):
        # `not_delivered` carries "deliver". Read as a good outcome, the
        # report leads with the group it happens to LEAST — which is
        # exactly the wrong group.
        verdict = not verdict
    return verdict


def _keywords_in(words: List[str]) -> tuple:
    """The longest bad and good keyword this name actually uses.

    Matched by word stem, never by substring. Substring matching read
    "left" inside `leftover_stock` and called a stock column an attrition
    outcome — the same fault that once made the chat parser answer a
    question about income with the mean of `Age`, found inside the word
    "average".

    A stem still matches its inflections, which is the point: `churned`
    starts with "churn", `cancelled` with "cancel", `attrition` with
    "attrit". Keywords holding an underscore (`no_show`, `on_time`) are
    matched against the whole name, since they span two words.
    """
    spaced = "_".join(words)

    def longest(keywords):
        hits = [k for k in keywords
                if ("_" in k and k in spaced)
                or any(_is_inflection(w, k) for w in words)]
        return max(hits, key=len, default="")

    return longest(_BAD), longest(_GOOD)


# What may follow a keyword and still be the same word. Bare prefix
# matching is not enough: `leftover_stock` starts with "left", and a
# stock column reported as an attrition outcome is worse than one not
# reported at all.
_INFLECTIONS = frozenset((
    "", "s", "d", "ed", "es", "ing", "er", "ers", "ion", "ions", "ance",
    "led", "ling", "ment", "ments", "al", "ly", "ted", "ting", "ped",
    "ping", "ned", "ning", "y", "ies", "cy", "ure", "ual",
))


def _is_inflection(word: str, keyword: str) -> bool:
    """True when `word` is `keyword` or a grammatical form of it.

    "churned" is "churn"; "cancelled" is "cancel"; "attrition" is
    "attrit". "leftover" is not "left".
    """
    if not word.startswith(keyword):
        return False
    return word[len(keyword):] in _INFLECTIONS


def _is_negated(words: List[str], matched: str) -> bool:
    """True when the name says the *absence* of the thing it names.

    `matched` is the outcome word the polarity came from, and a negation
    inside it is not a negation of it: `no_show` is listed as a problem
    in its own right, and flipping it would report the group that shows
    up least as the one doing well.
    """
    return any(w in _NEGATIONS and w not in matched for w in words)


@dataclass(frozen=True)
class Outcome:
    """A binary column worth explaining, and how to talk about it."""
    column: str
    noun: str
    # True: the flag firing is the good result, so the WORST group leads.
    # False: it is the bad result, so the worst group leads too — from the
    # other end of the table. None: the name does not say, so the report
    # states the concentration and passes no judgement.
    good: Optional[bool]
    rate: float
    events: int

    @property
    def named(self) -> bool:
        return self.good is not None


def discover_outcomes(df: pd.DataFrame,
                      limit: int = MAX_OUTCOMES) -> List[Outcome]:
    """Binary outcome columns in this frame, best first.

    "Best" is the one with the most left to explain: a flag at 30% has
    two populated sides to compare, one at 2% is a rare-event problem
    that a rate table answers badly. A column whose name is recognised
    outranks one whose name is not, because the report can say more
    about it.
    """
    from app.engines.domains.base import is_id_column

    if df is None or len(df) < MIN_ROWS:
        return []

    found: List[Outcome] = []
    for column in df.columns:
        try:
            series = df[column]
            if is_id_column(column, series):
                continue
            # Two distinct values is the whole test — a flag arrives as
            # 0/1, True/False, Yes/No or Y/N depending on who exported
            # it, and binary_mask already knows all four.
            mask = binary_mask(series)
            if mask is None:
                continue
            events = int(mask.sum())
            rate = float(mask.mean() * 100)
            if not (MIN_RATE_PCT <= rate <= MAX_RATE_PCT):
                continue
            # Both sides need enough records for a group comparison to
            # survive the Bonferroni correction downstream.
            if min(events, len(mask) - events) < MIN_EVENTS:
                continue
            found.append(Outcome(column=str(column),
                                 noun=_noun_for(column),
                                 good=_direction(column),
                                 rate=round(rate, 2),
                                 events=events))
        except Exception:
            logger.debug("outcome check failed on %s", column, exc_info=True)

    # Closest to an even split first, and a named outcome ahead of an
    # anonymous one at the same balance.
    found.sort(key=lambda o: (o.good is None, abs(o.rate - 50)))
    return found[:limit]
