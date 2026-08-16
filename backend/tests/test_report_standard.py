"""
The house standard every report has to pass, whatever domain it is for.

"Big 4 standard" is not a look — a firm's reports are recognisable
because of what they refuse to do, not because of their cover page. The
rules below are the ones that actually separate a professional
deliverable from a generated one, and each is stated as something the
document must never contain, because that is what can be tested.

Every rule here has a specific defect behind it, found in this app:

  1. **Every claim carries a figure.** "Revenue distribution analysis —
     high variability detected" was a heading pretending to be a finding.
  2. **Nothing states a cause it did not establish.** A bar chart shows
     that North is higher; it never shows why.
  3. **Nothing predicts.** "Attrition will reach 25%" appeared in a
     report built from a cross-section with no time dimension.
  4. **No benchmark without a source.** "Target 4.0+" was three-quarters
     of the way up whatever scale the column used, presented as a
     commitment somebody had made.
  5. **Numbers are written, not printed.** `7.72e+04` opened the
     executive summary of a financial review.
  6. **Urgency is earned.** Recommendations were stamped CRITICAL by
     their position in a list.
  7. **The document does not argue with itself.** "Gross Margin Healthy"
     sat beside "Gross Margin Down 8 Points".
  8. **The tool is never named.** The person delivering the work signs
     it; the report never mentions what produced it.
  9. **Nothing is empty.** A section with a heading and no content reads
     as a page that failed to load.

Run against every domain, because a standard that holds only for the
domain it was written against is not a standard.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  One realistic file per domain
# ══════════════════════════════════════════════════════════

def _finance():
    rng = np.random.default_rng(5)
    rows = []
    for i, m in enumerate(pd.date_range("2023-01-31", periods=30, freq="ME")):
        for cc in ("Retail", "Wholesale", "Services", "Support"):
            rev = rng.normal(7e5, 5e4) * (1 + i * 0.015)
            cogs = rev * (0.60 + i * 0.003)
            rows.append({"period": m, "cost_centre": cc,
                         "revenue": round(rev, 2), "cogs": round(cogs, 2),
                         "gross_profit": round(rev - cogs, 2),
                         "opex": round(rev * .18, 2),
                         "budget": round(rev * 1.06, 2)})
    return pd.DataFrame(rows)


def _hr():
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Eng", "Ops"], n),
        "job_title": rng.choice(["Analyst", "Manager"], n),
        "salary": rng.normal(60_000, 12_000, n).round(),
        "tenure_years": rng.gamma(1.6, 1.9, n).round(1),
        "manager_id": rng.integers(1, 40, n),
        "satisfaction": rng.integers(1, 6, n),
    })
    df["attrition"] = np.where(
        (df.tenure_years < 2) & (rng.random(n) < .5), "Yes", "No")
    return df


def _sales():
    rng = np.random.default_rng(3)
    n = 900
    return pd.DataFrame({
        "opportunity_id": range(n),
        "created_date": pd.to_datetime("2024-01-01")
                        + pd.to_timedelta(rng.integers(0, 540, n), "D"),
        "sales_rep": rng.choice(list("ABCDE"), n),
        "territory": rng.choice(["EMEA", "AMER", "APAC"], n),
        "deal_stage": rng.choice(
            ["Prospect", "Qualified", "Closed Won", "Closed Lost"], n,
            p=[.30, .25, .25, .20]),
        "deal_amount": rng.lognormal(9.5, .8, n).round(2),
        "quota": 250_000.0,
        "product_line": rng.choice(["Core", "Plus"], n),
    })


def _ecommerce():
    rng = np.random.default_rng(9)
    n = 2_000
    cat = rng.choice(["Home", "Beauty", "Electronics"], n)
    price = {"Home": 45, "Beauty": 22, "Electronics": 260}
    df = pd.DataFrame({
        "order_id": rng.integers(1, 800, n),
        "customer_id": rng.integers(1, 400, n),
        "order_date": pd.to_datetime("2024-01-01")
                      + pd.to_timedelta(rng.integers(0, 400, n), "D"),
        "product_sku": [f"S{i:03d}" for i in rng.integers(1, 200, n)],
        "category": cat,
        "unit_price": [round(rng.normal(price[c], price[c] * .2), 2)
                       for c in cat],
        "quantity": rng.integers(1, 5, n),
        "rating": rng.choice([1, 2, 3, 4, 5], n),
        "country": rng.choice(["IN", "US", "UK"], n),
    })
    df["revenue"] = (df.unit_price * df.quantity).round(2)
    return df


FILES = {"finance": _finance(), "hr": _hr(),
         "sales": _sales(), "ecommerce": _ecommerce()}
DOMAINS = sorted(FILES)


def _story(name):
    from app.engines.story_engine import generate_story

    return generate_story(FILES[name])


def _prose(name):
    """Everything in the report a client actually reads."""
    story = _story(name)
    parts = [story.headline, story.executive_summary]
    parts += story.key_findings + story.business_risks
    parts += story.opportunities + story.recommended_actions
    for ins in story.top_insights:
        parts += [ins.title, ins.problem, ins.cause, ins.evidence,
                  ins.action, ins.impact]
    return " ".join(p for p in parts if p)


# ══════════════════════════════════════════════════════════
#  1. Every claim carries a figure
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_every_finding_carries_a_number(domain):
    """"High variability detected" is a heading pretending to be a
    finding."""
    bare = [f for f in _story(domain).key_findings
            if not any(ch.isdigit() for ch in f)]
    assert not bare, bare


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_insight_states_its_evidence(domain):
    for ins in _story(domain).top_insights:
        assert ins.evidence.strip(), ins.title
        assert any(ch.isdigit() for ch in ins.evidence), ins.title


# ══════════════════════════════════════════════════════════
#  2. Nothing states a cause it did not establish
# ══════════════════════════════════════════════════════════

_CAUSAL = ("because of", "caused by", "is driving", "due to the",
           "as a result of", "leads to", "resulted in")

# A sentence that *denies* a causal claim is the opposite of the defect:
# "what is driving the movement is not identifiable from this data" is
# exactly the hedge the standard asks for. Matching the bare phrase
# flagged the hedges along with the assertions.
_DENIALS = ("not identifiable", "cannot", "is not", "not yet", "not "
            "proven", "unclear", "not separable", "not measured",
            "not established", "no way to")


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_unhedged_causal_claim(domain):
    for sentence in re.split(r"(?<=[.!?])\s+", _prose(domain)):
        low = sentence.lower()
        if any(d in low for d in _DENIALS):
            continue
        for phrase in _CAUSAL:
            assert phrase not in low, (phrase, sentence[:160])


# ══════════════════════════════════════════════════════════
#  3. Nothing predicts
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_nothing_is_stated_as_a_future_fact(domain):
    """A cross-section cannot support a forecast. Anything that reads
    like one is an assertion the data does not carry."""
    prose = _prose(domain)
    futures = re.findall(r"\bwill\s+(?:be|reach|rise|fall|grow|drop|"
                         r"increase|decrease|continue|become)\b",
                         prose, flags=re.I)
    assert not futures, futures


@pytest.mark.parametrize("domain", DOMAINS)
def test_projections_are_labelled_as_arithmetic(domain):
    """Where the report quantifies an upside it must be the arithmetic
    of closing a measured gap, not a forecast."""
    for ins in _story(domain).top_insights:
        if "worth about" in ins.impact or "is worth" in ins.impact:
            assert any(w in ins.impact.lower() for w in
                       ("measured", "arithmetic", "upper bound",
                        "not a forecast", "against their own",
                        "coachable")), ins.impact


# ══════════════════════════════════════════════════════════
#  4. No benchmark without a source
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_no_invented_target(domain):
    """"Target 4.0+" was a position on a scale presented as a
    commitment somebody had made."""
    prose = _prose(domain)
    invented = re.findall(r"\(Target [\d.]+\+?\)", prose)
    assert not invented, invented


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_cited_range_names_who_says_so(domain):
    prose = _prose(domain)
    if "guidance range" in prose:
        assert ("general guidance" in prose or "SHRM" in prose
                or "internal" in prose), "a range with no attribution"


# ══════════════════════════════════════════════════════════
#  5. Numbers are written, not printed
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_no_scientific_notation(domain):
    found = re.findall(r"\d\.?\d*e[+-]\d+", _prose(domain))
    assert not found, found


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_raw_float_noise(domain):
    """"18420.000000001" is a float, not a figure."""
    found = re.findall(r"\d+\.\d{5,}", _prose(domain))
    assert not found, found


# ══════════════════════════════════════════════════════════
#  6. Urgency is earned
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_critical_actions_need_a_critical_finding(domain):
    story = _story(domain)
    if not story.critical_issues:
        flagged = [a for a in story.recommended_actions if "[CRITICAL]" in a]
        assert not flagged, flagged


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_severity_ladder_is_not_all_one_value(domain):
    """Everything marked HIGH is the same as nothing being marked."""
    levels = {i.severity for i in _story(domain).top_insights}
    assert len(levels) >= 2 or len(_story(domain).top_insights) <= 2, levels


# ══════════════════════════════════════════════════════════
#  7. The document does not argue with itself
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_no_finding_contradicts_another(domain):
    titles = " | ".join(i.title for i in _story(domain).top_insights)
    contradictions = [
        ("Margin Healthy", "Margin Down"),
        ("Targets Exceeded", "Target Gap"),
        ("Healthy", "Critical"),
    ]
    for a, b in contradictions:
        assert not (a in titles and b in titles), (a, b, titles)


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_headline_is_not_counted_twice(domain):
    """The lead finding was counted again as "1 additional warning" in
    the next sentence.

    "Additional" only has to exclude the headline when the headline *is*
    a warning. Where the lead is a critical finding or an attrition
    figure, every warning genuinely is additional, and the count should
    match — the earlier version of this test asserted otherwise and was
    wrong about three domains out of four."""
    story = _story(domain)
    stated = re.search(r"(\d+) additional warning", story.executive_summary)
    if not stated:
        return
    warnings = sum(1 for i in story.top_insights if i.severity == "warning")
    lead_is_warning = bool(story.top_insights) and \
        story.top_insights[0].severity == "warning" and \
        story.top_insights[0].problem[:40] in story.executive_summary
    expected = warnings - 1 if lead_is_warning else warnings
    assert int(stated.group(1)) <= max(expected, 0), \
        (story.executive_summary, warnings, lead_is_warning)


# ══════════════════════════════════════════════════════════
#  8. The tool is never named
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_no_model_or_vendor_is_named(domain):
    prose = _prose(domain).lower()
    for name in ("claude", "anthropic", "openai", "gpt", "gemini",
                 "llm", "language model", "ai-generated", "chatgpt"):
        assert name not in prose, (name, domain)


# ══════════════════════════════════════════════════════════
#  9. Nothing is empty
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", DOMAINS)
def test_the_report_has_something_to_say(domain):
    story = _story(domain)
    assert story.executive_summary.strip()
    assert story.key_findings
    assert story.top_insights
    assert story.recommended_actions


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_placeholder_text_survives(domain):
    prose = _prose(domain).lower()
    for placeholder in ("n/a —", "todo", "lorem", "xxx", "tbd",
                        "placeholder", "insert "):
        assert placeholder not in prose, (placeholder, domain)


# ══════════════════════════════════════════════════════════
#  The rendered document
# ══════════════════════════════════════════════════════════

def _pdf(domain):
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf

    df = FILES[domain]
    story = _story(domain)
    return build_pdf(
        df=df,
        config={"title": "Review", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=story.top_insights, attrition=story.attrition,
        domain=story.domain)


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_document_states_its_basis_of_preparation(domain):
    """A figure a reader cannot trace is a figure they cannot use."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(_pdf(domain)))
    text = " ".join((doc[i].get_textpage().get_text_range() or "")
                    for i in range(len(doc)))
    assert "Methodology" in text or "Basis of Preparation" in text


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_page_is_nearly_blank(domain):
    """A heading with no content under it reads as a page that failed to
    load."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(_pdf(domain)))
    thin = []
    for i in range(1, len(doc)):
        text = (doc[i].get_textpage().get_text_range() or "").strip()
        # A dashboard page is mostly images and carries little text.
        if "the question it answers" in text:
            continue
        if 0 < len(text) < 200:
            thin.append(i)
    assert not thin, thin


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_document_carries_the_dashboard(domain):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(_pdf(domain)))
    text = " ".join((doc[i].get_textpage().get_text_range() or "")
                    for i in range(len(doc)))
    assert "the question it answers" in text, domain
