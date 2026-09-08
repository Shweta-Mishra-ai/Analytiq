"""
Five verticals added after the original eight — and the test that they
are engines rather than labels.

The registry exists because a domain used to be able to be *detectable*
without being *analysable*: marketing scored 0.91 on marketing data and
was then handed to the general engine, so the report carried a marketing
heading over generic column statistics. The completeness test in
test_domain_registry.py stops a domain registering without KPIs, a
blueprint and its own prompts. It cannot tell whether the engine behind
it says anything.

So each case here plants a fact a practitioner in that field would open
the review with, and asserts the engine finds it. A domain that detects
correctly and reports nothing specific fails, which is the outcome worth
protecting against.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains.registry import REGISTRY, detect_domain
from app.engines.story_engine import generate_story

N = 1200


def _r():
    return np.random.default_rng(20260908)


def education_frame():
    r = _r()
    module = r.choice(["Statistics", "Marketing", "Accounting", "Law"], N,
                      p=[0.3, 0.25, 0.25, 0.2])
    attendance = np.clip(r.normal(82, 14, N), 20, 100)
    grade = np.clip(35 + attendance * 0.45 - (module == "Statistics") * 14
                    + r.normal(0, 8, N), 0, 100)
    return pd.DataFrame({
        "student_id": range(1, N + 1),
        "module": module,
        "semester": r.choice(["2025-S1", "2025-S2"], N),
        "attendance_rate": attendance.round(1),
        "study_hours": r.integers(2, 30, N),
        "exam_score": grade.round(1),
        "passed": (grade >= 50).astype(int),
    })


def logistics_frame():
    r = _r()
    carrier = r.choice(["FastFreight", "BudgetHaul", "PrimeLogix"], N,
                       p=[0.6, 0.25, 0.15])
    late_p = np.where(carrier == "BudgetHaul", 0.42, 0.06)
    transit = np.where(carrier == "BudgetHaul", r.exponential(6, N) + 2,
                       r.normal(3, 0.6, N))
    return pd.DataFrame({
        "shipment_id": range(1, N + 1),
        "carrier": carrier,
        "lane": r.choice(["North-South", "East-West", "Coastal"], N),
        "transit_days": np.clip(transit, 1, 40).round(1),
        "freight_cost": (r.uniform(200, 900, N) *
                         np.where(carrier == "PrimeLogix", 2.4, 1.0)).round(2),
        "weight_kg": r.integers(50, 5000, N),
        "distance_km": r.integers(80, 2200, N),
        "on_time": (r.random(N) > late_p).astype(int),
    })


def realestate_frame():
    r = _r()
    loc = r.choice(["Riverside", "Old Town", "Northgate"], N,
                   p=[0.35, 0.3, 0.35])
    sqft = r.integers(450, 3000, N)
    ppsf = np.where(loc == "Old Town", 620,
                    np.where(loc == "Riverside", 410, 300))
    dom = np.where(loc == "Northgate", r.integers(70, 220, N),
                   r.integers(10, 70, N))
    return pd.DataFrame({
        "listing_id": range(1, N + 1),
        "location": loc,
        "property_type": r.choice(["Flat", "House", "Maisonette"], N),
        "bedrooms": r.integers(1, 6, N),
        "sqft": sqft,
        "price": (sqft * ppsf * r.uniform(0.9, 1.1, N)).round(0),
        "days_on_market": dom,
        "occupancy": np.clip(r.normal(88, 7, N), 40, 100).round(1),
        "sold": (r.random(N) < np.where(loc == "Northgate", 0.3, 0.7))
                .astype(int),
    })


def insurance_frame():
    r = _r()
    product = r.choice(["Motor", "Home", "Travel"], N, p=[0.45, 0.35, 0.2])
    premium = np.where(product == "Motor", r.uniform(400, 1400, N),
                       np.where(product == "Home", r.uniform(200, 700, N),
                                r.uniform(40, 180, N)))
    lr = np.where(product == "Travel", 1.5,
                  np.where(product == "Motor", 0.72, 0.55))
    claimed = (r.random(N) < 0.12).astype(int)
    claim = np.where(claimed == 1,
                     premium * lr / 0.12 * r.uniform(0.4, 1.8, N), 0)
    return pd.DataFrame({
        "policy_number": range(1, N + 1),
        "product": product,
        "region": r.choice(["North", "South", "Central"], N),
        "premium": premium.round(2),
        "sum_insured": (premium * r.uniform(20, 60, N)).round(0),
        "claimed": claimed,
        "claim_amount": claim.round(2),
        "lapsed": (r.random(N) < 0.11).astype(int),
    })


def energy_frame():
    r = _r()
    site = r.choice(["Plant A", "Plant B", "Office"], N, p=[0.4, 0.35, 0.25])
    hour = r.integers(0, 24, N)
    working = (hour >= 7) & (hour <= 19)
    base = np.where(site == "Office", 40, 140)
    load = base * np.where(working, r.uniform(2.5, 4.5, N),
                           r.uniform(0.5, 0.7, N))
    load = load * np.where(site == "Plant B", 2.0, 1.0)
    area = np.where(site == "Office", 2000,
                    np.where(site == "Plant A", 9000, 8500))
    return pd.DataFrame({
        "meter_id": range(1, N + 1),
        "site": site,
        "hour": hour,
        "consumption_kwh": load.round(1),
        "floor_area": area,
        "emissions_co2e": (load * 0.21).round(2),
        "energy_cost": (load * 0.28).round(2),
    })


# domain -> (frame builder, what a practitioner must see in the report)
CASES = {
    # The weak module, and the attendance effect behind it.
    "education": (education_frame, ("statistics", "attendance")),
    # The carrier that misses its dates, and the one that costs 2.4x.
    "logistics": (logistics_frame, ("budgethaul", "primelogix")),
    # The expensive location per square foot, and the illiquid one.
    "realestate": (realestate_frame, ("old town", "northgate")),
    # The product losing money, named, with the loss ratio attached.
    "insurance": (insurance_frame, ("travel", "loss ratio")),
    # A peak that drives capacity charges, and the inefficient site.
    "energy": (energy_frame, ("peak", "plant b")),
}


@pytest.mark.parametrize("domain", sorted(CASES))
def test_the_vertical_is_detected(domain):
    df = CASES[domain][0]()
    detected, confidence = detect_domain(df)
    assert detected == domain, \
        "{} data routed to {} — its engine never runs".format(
            domain, detected)
    assert confidence > 0.3


@pytest.mark.parametrize("domain", sorted(CASES))
def test_the_vertical_finds_what_was_planted(domain):
    """The check that separates an engine from a label."""
    build, expected = CASES[domain]
    story = generate_story(build())
    blob = " ".join(
        [i.title + " " + i.problem + " " + i.evidence
         for i in (list(story.critical_issues) + list(story.top_insights)
                   + list(story.positive_findings))]
        + list(story.key_findings) + list(story.business_risks)
        + list(story.opportunities)).lower()
    missing = [e for e in expected if e not in blob]
    assert not missing, "{}: report never mentions {}".format(domain, missing)


@pytest.mark.parametrize("domain", sorted(CASES))
def test_the_vertical_proposes_something(domain):
    """A report that names a problem and proposes nothing is half a
    report."""
    story = generate_story(CASES[domain][0]())
    assert story.recommended_actions or story.opportunities, \
        "{}: no action and no opportunity".format(domain)


@pytest.mark.parametrize("domain", sorted(CASES))
def test_the_vertical_speaks_its_own_language(domain):
    """Registered but generic is the failure this whole registry exists
    to prevent, so each engine has to produce at least one insight
    tagged to its own domain rather than to the shared passes."""
    story = generate_story(CASES[domain][0]())
    categories = {i.category for i in
                  (list(story.critical_issues) + list(story.top_insights)
                   + list(story.positive_findings))}
    # "<domain>_concentration" is the shared outcome pass, which runs for
    # every domain from one dispatch point. Excluding it is the whole
    # point of this test: a domain whose only domain-tagged insight comes
    # from shared machinery is exactly the relabel the registry exists to
    # prevent.
    own = [c for c in categories
           if c.startswith(domain) and not c.endswith("_concentration")]
    assert own, "{}: every insight came from a shared pass — {}".format(
        domain, sorted(categories))


def test_the_new_verticals_are_all_registered():
    for domain in CASES:
        assert domain in REGISTRY
        assert REGISTRY[domain].signature
        assert REGISTRY[domain].kpis
