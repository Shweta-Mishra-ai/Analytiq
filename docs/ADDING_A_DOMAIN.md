# Adding a domain

Analytiq detects the kind of business a dataset describes and routes it to
a matching insight engine. This is how you add a new one.

Adding a domain is **two files**: the engine, and one registration entry.
Nothing in the frontend changes — the UI renders whatever domain string the
API returns, and contains no hardcoded domain names.

## Why it works this way

Domain knowledge used to live in seven places that had to be edited in
lockstep: detection keywords, an `if/elif` dispatch, report blueprints,
industry benchmarks, PDF themes, and the narrator's prompt and label maps.
They drifted, silently:

- `marketing` was detectable — it scored 0.91 on marketing data — but had
  no insight engine, so the dispatch `else` handed it to the general
  engine. The report said "marketing" at the top and then described
  generic column statistics.
- Benchmarks existed for `saas`, `operations` and `healthcare` that
  nothing could reach, because detection had never heard of those domains.

Neither failed. Both just produced a worse report. `DomainSpec` collects
the facets in one place and `tests/test_domain_registry.py` fails if any
registered domain is missing one, so this class of bug cannot come back.

## Step 1 — write the engine

Create `backend/app/engines/domains/<name>.py` exporting one function:

```python
def _insights_<name>(df, stats, corrs) -> dict:
    return {"findings": [...],       # list[str], the headline facts
            "risks": [...],          # list[str]
            "opportunities": [...],  # list[str]
            "actions": [...],        # list[str]
            "insights": [...]}       # list[Insight]
```

Build each `Insight` with `build_insight()` from `domains/base.py`. It
takes `title, problem, cause, evidence, action, impact, severity,
category` — every field is required, and the completeness tests check none
comes back empty.

`domains/_common.py` has the helpers every engine needs: `find_measure`
(locate a numeric column by concept, never an identifier), `binary_rate`,
`segment_gap`, `concentration`, `variability`, `fmt`, and
`benchmark_note`.

Anything the engine presents as a finding should pass through
`app/engines/rigour.py` first. A comparison needs both significance and a
usable effect size (`assess_finding`), and a model has to beat the
obvious guess before its drivers mean anything (`assess_classifier`,
`assess_regressor`). A domain engine that reports a difference the gate
would reject is reporting noise.

### KPI specs

```python
kpis=(
    K("attrition", "Attrition Rate", "rate",
      ("attrition", "left", "exited"), unit="%",
      benchmark="attrition_rate", higher_is_better=False),
    K("median_pay", "Median Salary", "median", ("salary", "income")),
)
```

`kind` is one of `count`, `sum`, `mean`, `median`, `rate`, `ratio`,
`nunique`. Choose it for the metric, not for convenience: a total of a
1-5 rating means nothing, and a total of an identifier means less. That
is what the panel used to do — "Σ EmployeeNumber" on an HR extract. A KPI
whose columns are not present is omitted rather than shown empty.

Two rules the tests enforce:

- **Stay in your domain's vocabulary.** A marketing report must not say
  "attrition"; an operations report must not say "churn". See
  `FOREIGN_VOCAB` in `tests/test_expansion_domains.py`.
- **Survive degenerate input.** One row, all-NaN, and a single-group frame
  must return the dict shape without raising. A domain engine must never
  be what breaks a report.

## Step 2 — register it

In `backend/app/engines/domains/registry.py`:

```python
register(DomainSpec(
    key="<name>",
    label="<Prose Label>",      # "SaaS", not "saas", where it's an acronym
    signature=(...),            # distinctive words — weight 3
    keywords=(...),             # supporting words — weight 1
    insight_fn=_insights_<name>,
    pdf_theme="Corporate Light",   # must exist in pdf_builder.THEMES
))
```

**Signature vs keywords.** A signature word should be decisive: `mrr`,
`readmission`, `cycletime`. A keyword is shared vocabulary that means
little alone: `revenue`, `cost`, `department`. Signature words must be
unique across domains — a test enforces it, because a word claimed by two
domains cannot decide between them.

Detection splits camelCase, so `MonthlyCharges` matches `monthly` and
`charges`. Keywords match a whole word, or a substring of a column name
when at least 5 characters long.

A domain is only claimed when it clears `MIN_SIGNAL` (6.0 weighted) *and*
beats the runner-up by `MIN_MARGIN` (1.35x). Otherwise detection returns
`general`. This is deliberate: a wrong domain is worse than no domain,
because it routes the data to an engine that speaks the wrong language and
prints confident numbers about the wrong thing.

## Step 3 — the facets the tests require

`test_every_domain_is_complete` will fail until all of these exist:

| Facet | Where | Notes |
|---|---|---|
| Report blueprint | `report_blueprints.py` → `BLUEPRINTS` | Section categories must match the `category` values your engine emits |
| Benchmarks | `industry_benchmarks.py` → `DOMAIN_BENCHMARKS` | Plus a `_COLUMN_KEYWORD_MAP` entry so lookups resolve |
| Executive prompt | `ai/prompt_builder.py` → `EXECUTIVE_PROMPTS` | |
| Insight prompt | `ai/prompt_builder.py` → `INSIGHT_PROMPTS` | |
| PDF theme | `pdf_builder.py` → `THEMES` | Reuse an existing theme unless the domain needs its own |
| KPI specs | `DomainSpec.kpis` | The five or six numbers a reader of this data looks for first |
| Chart metrics | `DomainSpec.chart_metrics` | Column fragments this domain would rather chart |

Prompts fall back to `GENERAL_*`, never to another domain's — the old
`.get(domain, HR_EXECUTIVE_PROMPT)` default put "employees" and
"attrition" into finance summaries.

## Step 4 — test it

Add your domain to `BUILDERS` and `FOREIGN_VOCAB` in
`tests/test_expansion_domains.py` with a realistic sample frame. Build
something the engine should genuinely find — the marketing fixture makes
one channel ~6x less efficient so the waste rule has a real target.

Also add a fixture to `BUILDERS` in
`tests/test_end_to_end_matrix.py`. That file runs every engine and a full
report build against every registered domain, and
`test_every_registered_domain_has_a_matrix_fixture` fails if you register
a domain without covering it there — the original defects survived
precisely because nothing exercised the whole pipeline per domain.

```bash
cd backend
python -m pytest tests/test_domain_registry.py \
                tests/test_expansion_domains.py \
                tests/test_end_to_end_matrix.py -q
python -m pytest tests/ -q      # full suite before you commit
```

## Special scope rules

If a domain carries a boundary on what the analysis may claim, write it
into the engine, the blueprint's `reference_note`, and the prompts —
all three. `healthcare` is the worked example: it is administrative
analysis only, emits no clinical guidance, and reasons about no individual
patient.

## Checklist

- [ ] `domains/<name>.py` with `_insights_<name>(df, stats, corrs)`
- [ ] `DomainSpec` registered, signature words unique
- [ ] Blueprint, benchmarks, both prompts, valid PDF theme
- [ ] Fixture added to `test_expansion_domains.py`
- [ ] Fixture added to `test_end_to_end_matrix.py`
- [ ] Optional: a PDF deep page, attached with `attach_deep_page`
- [ ] `kpis` and `chart_metrics` on the spec
- [ ] Full suite green
