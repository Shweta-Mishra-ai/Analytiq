"""
engines/dashboard_spec.py — which tiles a dashboard should carry, for
the kind of business this data is about.

Every dataset got the same five tiles: a line, a bar, a pie, a histogram
and a correlation matrix. That is a chart grid, not a dashboard. A
finance director opening a P&L expects margin, cost structure and budget
variance; an HR director expects headcount, attrition by department and
tenure; a sales director expects the funnel and quota attainment. One
fixed set serves none of them, and the reader can tell in two seconds
that nobody decided what this file was about.

Three rules hold this together.

**A tile has to answer a question.** Each one here carries the question
it answers, in the language of the function. If a tile cannot be
phrased as something a person in that role would ask, it does not belong
on the page.

**The mark follows the question, not the column type.** Composition is a
donut, a trend is a line, a comparison across a handful of groups is a
bar, a relationship between two measures is a scatter, a spread is a
histogram, and geography is a map. Reaching for a pie because there
happened to be a categorical column is how a chart pack becomes noise.

**A tile the data cannot support is dropped, not padded.** Every entry
declares what it needs; if the columns are not there, the tile does not
appear and nothing takes its place. Six tiles that mean something beat
twenty that fill a grid — which is the whole difference between a
dashboard and a screenshot of a spreadsheet.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.engines.column_roles import Roles, resolve

logger = logging.getLogger(__name__)

# A dashboard is a page someone reads in a meeting. Past about eight
# tiles it stops being read and starts being scrolled.
MAX_TILES = 8


@dataclass
class Tile:
    """One tile, and the question it exists to answer."""
    type: str
    title: str
    question: str
    x: Optional[str] = None
    y: Optional[str] = None
    # The second series, for a comparison tile. Actual against plan is a
    # headline finance question and it cannot be asked with one measure.
    y2: Optional[str] = None
    agg: str = "sum"
    w: int = 6
    h: int = 5
    # Lower sorts first. The lead measure gets the top of the page.
    priority: int = 50

    def as_dict(self, index: int) -> Dict:
        return {"id": "t{}".format(index), "type": self.type, "x": self.x,
                "y": self.y, "y2": self.y2, "agg": self.agg,
                "title": self.title, "question": self.question,
                "w": self.w, "h": self.h}


@dataclass
class Candidate:
    """A tile the domain wants, and what it needs to exist."""
    needs: tuple                       # role names that must resolve
    build: Callable[[Roles, pd.DataFrame], Optional[Tile]]
    # Extra guard beyond the roles — cardinality, row count, and so on.
    when: Optional[Callable[[Roles, pd.DataFrame], bool]] = None


def _cardinality(df: pd.DataFrame, col: Optional[str]) -> int:
    if not col or col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


# ══════════════════════════════════════════════════════════
#  Shared tiles
# ══════════════════════════════════════════════════════════

def _trend(label: str, question: str, role: str = "money", priority: int = 10):
    def build(r: Roles, _df):
        return Tile("line", label.format(r.get(role)), question,
                    x=r.period, y=r.get(role), w=12, h=5, priority=priority)
    return Candidate(("period", role), build)


def _by_group(group_role: str, measure_role: str, label: str, question: str,
              priority: int = 20, w: int = 6):
    def build(r: Roles, df):
        group = r.get(group_role)
        n = _cardinality(df, group)
        # A bar chart of forty categories is a barcode. Past twelve the
        # comparison is better made as a ranked top-N, which is what the
        # builder does anyway — but the tile gets more width to do it.
        return Tile("bar", label.format(group=group, measure=r.get(measure_role)),
                    question, x=group, y=r.get(measure_role),
                    w=12 if n > 12 else w, h=5, priority=priority)
    return Candidate((group_role, measure_role), build,
                     when=lambda r, df: 2 <= _cardinality(df, r.get(group_role)) <= 40)


def _share(group_role: str, measure_role: str, label: str, question: str,
           priority: int = 30):
    def build(r: Roles, _df):
        return Tile("pie", label.format(group=r.get(group_role),
                                        measure=r.get(measure_role)),
                    question, x=r.get(group_role), y=r.get(measure_role),
                    w=5, h=5, priority=priority)
    # Composition only reads at a handful of slices. Above eight the eye
    # cannot compare them and a ranked bar is the honest chart.
    return Candidate((group_role, measure_role), build,
                     when=lambda r, df: 2 <= _cardinality(df, r.get(group_role)) <= 8)


def _relationship(a_role: str, b_role: str, label: str, question: str,
                  priority: int = 60):
    def build(r: Roles, _df):
        return Tile("scatter", label.format(a=r.get(a_role), b=r.get(b_role)),
                    question, x=r.get(a_role), y=r.get(b_role),
                    w=6, h=5, priority=priority)
    # A scatter needs enough points to show a shape, and two genuinely
    # different measures — plotting revenue against itself under another
    # name is a diagonal line.
    return Candidate((a_role, b_role), build,
                     when=lambda r, df: (len(df) >= 30
                                         and r.get(a_role) != r.get(b_role)))


def _spread(role: str, label: str, question: str, priority: int = 70):
    def build(r: Roles, _df):
        return Tile("histogram", label.format(measure=r.get(role)), question,
                    x=r.get(role), w=6, h=4, priority=priority)
    return Candidate((role,), build, when=lambda r, df: len(df) >= 30)


# ══════════════════════════════════════════════════════════
#  Per-domain dashboards
# ══════════════════════════════════════════════════════════

FINANCE: List[Candidate] = [
    _trend("{} over time", "Is the top line moving, and in which direction?", "money", 10),
    # A P&L is cut by cost centre, not by product line — `unit` is the
    # organisational grouping, and pointing this at `product` left the
    # finance dashboard with nothing to break the numbers down by.
    _by_group("unit", "profit", "{measure} by {group}",
              "Which parts of the business earn, and which consume?", 20),
    _by_group("product", "profit", "{measure} by {group}",
              "Which product lines earn, and which consume?", 22),
    Candidate(("plan", "money"),
              lambda r, df: Tile("comparison",
                                 "{} against {}".format(r.money, r.plan),
                                 "Where did we land against plan?",
                                 x=r.unit or r.product or r.period,
                                 y=r.money, y2=r.plan,
                                 w=7, h=5, priority=25),
              when=lambda r, df: bool(r.unit or r.product or r.period)),
    _share("unit", "cost", "Cost split by {group}",
           "Where does the money go?", 30),
    _share("product", "cost", "Cost split by {group}",
           "Which lines consume the cost base?", 32),
    _trend("{} over time", "Is the cost base tracking the top line?", "cost", 40),
    _spread("profit", "Spread of {measure}",
            "Is profitability consistent, or carried by a few periods?", 70),
]

HR: List[Candidate] = [
    Candidate(("attrition", "unit"),
              lambda r, df: Tile("bar", "Attrition rate by {}".format(r.unit),
                                 "Which parts of the organisation lose people?",
                                 x=r.unit, y=r.attrition, agg="mean",
                                 w=7, h=5, priority=10),
              when=lambda r, df: 2 <= _cardinality(df, r.unit) <= 30),
    Candidate(("unit",),
              lambda r, df: Tile("bar", "Headcount by {}".format(r.unit),
                                 "Where do the people sit?",
                                 x=r.unit, y=r.unit, agg="count",
                                 w=5, h=5, priority=15),
              when=lambda r, df: 2 <= _cardinality(df, r.unit) <= 30),
    _by_group("unit", "money", "Median {measure} by {group}",
              "Does pay differ between functions more than the roles "
              "explain?", 25),
    _spread("money", "Pay distribution",
            "How is pay spread, and is the average a fair summary?", 30),
    _by_group("region", "money", "Median {measure} by {group}",
              "Does pay differ by location more than the roles explain?", 40),
    _relationship("quantity", "money", "{a} against {b}",
                  "Does time served track pay?", 50),
    # A scatter of pay against a Yes/No flag is two vertical strips, not
    # a relationship. The comparison the question actually asks for is
    # the pay on each side of the flag.
    Candidate(("attrition", "money"),
              lambda r, df: Tile("bar", "Pay of leavers against stayers",
                                 "Are the people leaving the ones paid least?",
                                 x=r.attrition, y=r.money, agg="mean",
                                 w=5, h=4, priority=55),
              when=lambda r, df: len(df) >= 50),
]

SALES: List[Candidate] = [
    _trend("Bookings over time", "Is the pipeline converting at a steady rate?",
           "money", 10),
    _by_group("person", "money", "{measure} by {group}",
              "Who is carrying the number?", 15),
    _by_group("region", "money", "{measure} by {group}",
              "Where is performance concentrated?", 20),
    _by_group("unit", "money", "{measure} by {group}",
              "Which team is carrying the number?", 28),
    _share("product", "money", "{measure} share by {group}",
           "What are we actually selling?", 35),
    _spread("money", "Deal size distribution",
            "Is the total carried by a few large deals?", 45),
    _relationship("quantity", "money", "{a} against {b}",
                  "Do bigger deals take proportionally longer?", 60),
]

ECOMMERCE: List[Candidate] = [
    _trend("Revenue over time", "Is demand growing?", "money", 10),
    _by_group("product", "money", "{measure} by {group}",
              "Which categories carry the business?", 20),
    _share("region", "money", "{measure} share by {group}",
           "Where do orders come from?", 30),
    _spread("money", "Order value distribution",
            "What does a typical basket look like?", 40),
    _by_group("product", "rating", "Rating by {group}",
              "Where is the customer experience weakest?", 45),
    _relationship("rating", "money", "{a} against {b}",
                  "Do better-rated products actually sell more?", 55),
]

GENERAL: List[Candidate] = [
    _trend("{} over time", "Is the main measure moving?", "money", 10),
    _by_group("product", "money", "{measure} by {group}",
              "How does the measure differ across groups?", 20),
    _by_group("region", "money", "{measure} by {group}",
              "How does the measure differ by location?", 25),
    _share("product", "money", "{measure} share by {group}",
           "How concentrated is the total?", 35),
    _spread("money", "Spread of {measure}",
            "Where does the mass of the distribution sit?", 50),
    _relationship("quantity", "money", "{a} against {b}",
                  "Do the two measures move together?", 60),
]

DASHBOARDS: Dict[str, List[Candidate]] = {
    "finance": FINANCE,
    "hr": HR,
    "sales": SALES,
    "ecommerce": ECOMMERCE,
    "marketing": SALES,
    "operations": GENERAL,
    "saas": SALES,
    "healthcare": GENERAL,
    "general": GENERAL,
}


def _fallback_tiles(df: pd.DataFrame, roles: Roles) -> List[Tile]:
    """When the roles resolve to nothing, plot what is chartable.

    An empty dashboard is a worse answer than a generic one — but it is
    a better answer than a confident, wrong one, so this stays plain and
    says nothing about the business.
    """
    from app.engines.chart_engine import _cat_columns, rank_measures

    measures = rank_measures(df)
    cats = _cat_columns(df)
    dates = df.select_dtypes(include="datetime").columns.tolist()
    if not measures:
        return []
    lead = measures[0]
    tiles: List[Tile] = []
    if dates:
        tiles.append(Tile("line", "{} over time".format(lead),
                          "Is it moving?", x=dates[0], y=lead, w=12, h=5))
    if cats:
        tiles.append(Tile("bar", "{} by {}".format(lead, cats[0]),
                          "How does it differ across groups?",
                          x=cats[0], y=lead, w=7, h=5))
    if len(measures) > 1:
        tiles.append(Tile("histogram", "Spread of {}".format(measures[1]),
                          "Where does the mass sit?", x=measures[1],
                          w=5, h=5))
    return tiles


def build_spec(df: pd.DataFrame, domain: str = "general",
               roles: Optional[Roles] = None,
               max_tiles: int = MAX_TILES) -> List[Tile]:
    """The tiles this dataset can actually support, best first."""
    roles = roles or resolve(df)
    candidates = DASHBOARDS.get(str(domain or "").strip().lower(), GENERAL)

    tiles: List[Tile] = []
    for cand in candidates:
        if any(not roles.get(role) for role in cand.needs):
            continue
        try:
            if cand.when and not cand.when(roles, df):
                continue
            tile = cand.build(roles, df)
        except Exception:
            logger.debug("dashboard tile failed to build", exc_info=True)
            continue
        if tile and tile.x:
            tiles.append(tile)

    if not tiles:
        tiles = _fallback_tiles(df, roles)

    # Two tiles plotting the same measure against the same grouping is
    # the same tile twice — the HR and general sets can both want
    # "measure by product" when the roles overlap.
    seen = set()
    unique: List[Tile] = []
    for tile in sorted(tiles, key=lambda t: t.priority):
        key = (tile.type, tile.x, tile.y)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tile)

    return unique[:max_tiles]


def layout_tiles(tiles: List[Tile]) -> List[Dict]:
    """Place the tiles on a twelve-column grid, in reading order.

    Rows are filled. Packing on the declared widths alone left holes: a
    six-wide tile followed by a seven-wide one does not fit twelve
    columns, so the seven wrapped and the six sat next to white space.
    Half an empty row on a client-facing page reads as a chart that
    failed to load, so the last tile in each row is widened to close it.
    """
    rows: List[List[Tile]] = []
    current: List[Tile] = []
    used = 0
    for tile in tiles:
        if used + tile.w > 12 and current:
            rows.append(current)
            current, used = [], 0
        current.append(tile)
        used += tile.w
    if current:
        rows.append(current)

    out: List[Dict] = []
    index = 1
    row_top = 0
    for row in rows:
        slack = 12 - sum(t.w for t in row)
        if slack > 0:
            row[-1].w += slack
        col = 0
        for tile in row:
            spec = tile.as_dict(index)
            spec["gx"], spec["gy"] = col, row_top
            out.append(spec)
            col += tile.w
            index += 1
        row_top += max(t.h for t in row)
    return out
