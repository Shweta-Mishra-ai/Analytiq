"""
The columns a client actually uploads.

Real exports do not contain clean floats. A sales extract writes revenue
as "$1,200", a discount column as "15%", a finance export writes a loss
as "(500)", and a summary sheet writes "1.2k". pandas reads every one of
those as text, and Analytiq analysed only what pandas typed for it.

Measured on a five-column file of exactly that shape, the app found ONE
numeric column — `units` — and silently skipped revenue, discount and
margin. Nothing raised: every KPI, chart, correlation, driver and
forecast simply had nothing to work on, and the client received a report
about the one column that happened to be clean. That loses an account
without ever showing an error.

The second half of this file is about the error that would be worse than
not converting at all: "€1.850" is 1.85 in a US-formatted file and 1,850
in a German one, and choosing wrong understates revenue by a thousand
times while still looking plausible.
"""
import io

import pandas as pd
import pytest

from app.engines.data_loader import (_coerce_numeric_strings, _decimal_style,
                                     _looks_like_money_or_percent, load_file)


def _load(name: str, csv: str):
    raw = csv.encode()

    class Upload:
        size = len(raw)

        def __init__(self):
            self.name = name
            self._b = io.BytesIO(raw)

        def read(self, *a):
            return self._b.read(*a)

        def seek(self, *a):
            return self._b.seek(*a)

    return load_file(Upload())


CLIENT_CSV = (
    'order_id,revenue,discount,margin,units\n'
    'A1,"$1,200",15%,(500),3\n'
    'A2,"$2,450",8%,320,7\n'
    'A3,"$980",22%,(120),2\n'
    'A4,"$3,100",5%,890,9\n'
    'A5,"$1,750",12%,410,5\n'
    'A6,"$2,010",9%,150,6\n'
)


# ══════════════════════════════════════════════════════════
#  The columns become numbers
# ══════════════════════════════════════════════════════════

def test_a_real_client_export_is_not_reduced_to_one_column():
    """The defect that started this: four measures, one of them read."""
    df = _load("client.csv", CLIENT_CSV).df
    numeric = set(df.select_dtypes(include="number").columns)
    assert {"revenue", "discount", "margin", "units"} <= numeric


def test_the_values_are_right_not_merely_numeric():
    df = _load("client.csv", CLIENT_CSV).df
    assert df["revenue"].iloc[0] == 1200
    assert df["discount"].iloc[0] == 15      # as written, not 0.15
    assert df["margin"].iloc[0] == -500      # (500) is a loss


def test_the_change_is_reported_to_the_user():
    """Reading a column differently from how it was stored is a change to
    the client's data. They can only disagree if they are told."""
    res = _load("client.csv", CLIENT_CSV)
    joined = " ".join(res.warnings)
    assert "revenue" in joined
    assert "formatted text" in joined


@pytest.mark.parametrize("raw,expected", [
    ("$1,200",   1200),
    ("₹3,499",   3499),
    ("£920",      920),
    ("¥12,000", 12000),
    ("15%",        15),
    ("(500)",    -500),
    ("(1.2k)",  -1200),
    ("1.2k",     1200),
    ("3.4M", 3_400_000),
    ("1,234.56", 1234.56),
])
def test_the_shapes_a_finance_export_uses(raw, expected):
    got = _coerce_numeric_strings(pd.Series([raw]))[0]
    assert got == pytest.approx(expected)


def test_a_genuine_text_column_is_left_alone():
    """The check is on the data, not the column name — and words stay
    words. A guard that converts too eagerly destroys categories."""
    df = _load("text.csv",
               "region,band,note\n"
               "North,Low,ok\nSouth,High,late\nEast,Low,ok\n"
               "West,Medium,ok\nNorth,High,ok\nSouth,Low,late\n").df
    assert df.select_dtypes(include="number").columns.tolist() == []
    assert df["band"].tolist()[:3] == ["Low", "High", "Low"]


def test_identifier_columns_are_never_coerced():
    """An order id of "00123" must not become the number 123."""
    df = _load("ids.csv",
               "order_id,customer_code,revenue\n"
               '00123,C-001,"$100"\n00124,C-002,"$200"\n'
               '00125,C-003,"$300"\n00126,C-004,"$400"\n').df
    assert not pd.api.types.is_numeric_dtype(df["order_id"])


# ══════════════════════════════════════════════════════════
#  Which character is the decimal point
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("values,style", [
    (["$1,200", "$2,450", "$980"],        "us"),   # thousands
    (["12.5", "3.75", "108.2"],           "us"),   # decimals
    (["1.234,56", "2.480,10", "990,25"],  "eu"),   # both separators
    (["1.234.567", "2.000.100"],          "eu"),   # two dot groups
    (["15%", "8%", "22%"],                "us"),
])
def test_the_column_settles_its_own_decimal_separator(values, style):
    assert _decimal_style(values) == style


def test_european_amounts_parse_to_european_values():
    got = _coerce_numeric_strings(pd.Series(["1.234,56", "2.480,10"]), "eu")
    assert list(got) == pytest.approx([1234.56, 2480.10])


def test_an_ambiguous_column_is_refused_rather_than_guessed():
    """"€1.850" is 1.85 or 1,850 and the column does not say which.

    Converting on a guess would understate a revenue figure by a
    thousand times and still look plausible on the page — worse than
    leaving the column out, which at least shows up as a gap.
    """
    assert _decimal_style(["€1.850", "€2.300", "€990"]) == "unknown"

    res = _load("eu.csv",
                "sku,amount\n"
                "X1,€1.850\nX2,€2.300\nX3,€990\nX4,€1.400\nX5,€2.750\n")
    assert not pd.api.types.is_numeric_dtype(res.df["amount"])
    joined = " ".join(res.warnings)
    assert "amount" in joined
    assert "thousand times" in joined


def test_the_detector_itself_is_not_fooled_by_words():
    assert _looks_like_money_or_percent(pd.Series(["North", "South"])) is False
    assert _looks_like_money_or_percent(pd.Series(["$1,200", "$980"])) is True


# ══════════════════════════════════════════════════════════
#  What must never be mistaken for money
# ══════════════════════════════════════════════════════════
#
# The first version of the detector put "R$" (Brazilian real) into a
# character class, so the class quietly gained a bare "R" — and a Region
# column of R1, R2, R3 was read as the numbers 1, 2, 3. Coded categories
# are the columns every group-by in the product depends on; turning them
# into integers does not raise, it just silently analyses nonsense.

@pytest.mark.parametrize("values", [
    ["R1", "R2", "R3", "R10", "R4"],        # region codes
    ["Q1", "Q2", "Q3", "Q4"],               # quarters
    ["A1", "A2", "B1", "B2"],               # grades
    ["North", "South", "East", "West"],     # plain words
    ["2024-01", "2024-02", "2024-03"],      # period labels
])
def test_coded_categories_are_not_money(values):
    assert _looks_like_money_or_percent(pd.Series(values)) is False


@pytest.mark.parametrize("values", [
    ["$1,200", "$980", "$2,450"],
    ["R$ 1.200", "R$ 980", "R$ 2.450"],     # the real currency, unharmed
    ["15%", "8%", "22%"],
    ["1.2k", "3.4M", "900k"],
    ["(500)", "(120)", "(1,200)"],
    ["1,234", "5,600", "12,000"],
])
def test_formatted_money_is_still_recognised(values):
    assert _looks_like_money_or_percent(pd.Series(values)) is True


def test_a_region_column_survives_a_whole_load():
    """End to end, because the detector is only half the path."""
    res = _load("regions.csv",
                "region,revenue\n"
                'R1,"$1,200"\nR2,"$980"\nR3,"$2,450"\n'
                'R10,"$1,750"\nR4,"$3,100"\nR5,"$2,010"\n')
    df = res.df
    assert df["region"].tolist() == ["R1", "R2", "R3", "R10", "R4", "R5"]
    assert pd.api.types.is_numeric_dtype(df["revenue"])


def test_leading_zeros_survive_the_csv_reader():
    """pandas types "00123" as 123 while parsing, before any of our code
    sees the frame. A US zip of 01234 is Massachusetts; 1234 is nowhere.
    """
    res = _load("zips.csv",
                "order_id,zip,amount\n"
                '00123,01234,"$100"\n00124,02138,"$200"\n'
                '00125,90210,"$300"\n00126,10001,"$400"\n')
    df = res.df
    assert df["order_id"].astype(str).tolist()[0] == "00123"
    assert df["zip"].astype(str).tolist()[0] == "01234"
    assert " ".join(res.warnings).count("leading zeros") == 1
