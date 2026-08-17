# Reporting standard

This is the standard every report and dashboard Analytiq produces is
held to, and the checklist each one is signed off against before it
leaves the app.

It exists because "make the reports look like a Big 4 deliverable" is
not a design instruction. A firm's report is recognisable for what it
refuses to do — it does not assert a cause it has not established, does
not predict from a cross-section, does not cite a benchmark it cannot
attribute, and does not put a heading over an empty section. Those are
testable. A cover page is not.

Every rule below has a specific defect behind it, all of them found in
this codebase. Each is enforced in two places: at the point the text or
the chart is built, and again in a test that runs against four domains
on every commit. The rule numbers are the same in all three places —
here, in `backend/tests/test_report_standard.py`, and in the **Review
Checklist** printed in the appendix of every report.

---

## 1. Every claim carries a figure, and the figure is traceable

A finding without a number is a heading pretending to be a finding.
*"Revenue distribution analysis — high variability detected"* tells a
reader nothing they can act on or check.

Every finding contains a figure. Every insight carries an `evidence`
line naming the column and the count it came from, so a reader can go
back to the file and reproduce it.

- Built in: `story_engine`, the domain engines under `app/engines/domains/`
- Tested by: `test_every_finding_carries_a_number`,
  `test_every_insight_states_its_evidence`
- On the page: *Every finding carries a figure and names its source column*

## 2. No cause is asserted that the data cannot establish

A bar chart shows that North is higher than South. Nothing in it shows
why. *"Attrition rose because of the pay freeze"* is a claim about a
column the file does not contain.

Association is reported as association. Where a reader will inevitably
ask "why", the report says what would be needed to answer it — *"what
is driving this is not identifiable from the columns supplied; stage
entry dates would separate a slow stage from a fat one"*. The hedge is
required, not optional, and is not treated as a causal claim.

- Tested by: `test_no_unhedged_causal_claim`
- On the page: *No cause is asserted that the data cannot establish*

## 3. Nothing is stated as a future fact

*"Attrition will reach 25%"* appeared in a report built from a
cross-section with no time dimension at all.

Quantified upside is allowed, and is stated as what it is: the
arithmetic of closing a gap that has already been measured, marked
explicitly as *not a forecast*. Anything that reads as a prediction is
an assertion the data does not carry.

- Tested by: `test_nothing_is_stated_as_a_future_fact`,
  `test_projections_are_labelled_as_arithmetic`
- On the page: *Nothing is stated as a future fact*

## 4. Every threshold cited names who sets it

*"Average rating 3.6 (Target 4.0+)"* — nobody set that target. It was
three-quarters of the way up whatever scale the column happened to use,
printed as a commitment somebody had made.

A threshold is either attributed to the body that publishes it, or it is
described for what it is: a position within this dataset's own
distribution. Reference ranges come from the domain blueprint, so a
finance report cites finance conventions and an HR report cites HR
bodies — the single hardcoded list that put SHRM and Gallup in the
footer of every report regardless of subject is gone.

- Tested by: `test_no_invented_target`, `test_a_cited_range_names_who_says_so`
- On the page: *Every threshold cited names who sets it*

## 5. Figures are written for a reader, not printed by a machine

`7.72e+04` opened the executive summary of a financial review.
`18420.000000001` is a float, not a figure.

Money and volumes are written in the units a reader uses — `1.2m`,
`84k` — on the axis, in the labels and in the prose, and the same way in
all three so nobody has to convert between them to check that two
numbers agree.

- Built in: `app/services/numfmt.py`, used by the chart builders and the
  narrative engines
- Tested by: `test_no_scientific_notation`, `test_no_raw_float_noise`
- On the page: *Figures are written for a reader, not printed by a machine*

## 6. Urgency is earned

Recommendations were stamped `[CRITICAL]` by their position in a list.
Everything marked high is the same as nothing being marked.

Severity comes from the finding, not from the ordering. A critical
action requires a critical finding behind it, and a report whose
insights all carry one severity has not ranked anything.

- Tested by: `test_critical_actions_need_a_critical_finding`,
  `test_the_severity_ladder_is_not_all_one_value`
- On the page: *Urgency is earned, not assigned by position*

## 7. The document does not argue with itself

*"Gross Margin Healthy: 33.7%"* sat beside *"Gross Margin Down 8
Points"*. Both were true. Together they read as a report nobody had
read.

Where a level and a trend conflict, the trend wins and the level is
withdrawn — a margin falling eight points is the finding, and its
current value is context inside that finding rather than a second
headline.

- Tested by: `test_no_finding_contradicts_another`,
  `test_the_headline_is_not_counted_twice`
- On the page: *No finding contradicts another*

## 8. The preparer is accountable; the tooling is not named

A deliverable is signed by the person or firm responsible for the
conclusions. What produced the document is not the reader's concern and
appears nowhere in it — not in the methodology, not in the footer, not
as a watermark. `prepared_by` is the client's or the freelancer's name
and is theirs to set.

- Built in: `_prepared_by_line` in `app/engines/pdf_builder.py`
- Tested by: `test_no_model_or_vendor_is_named`
- On the page: *The preparer is accountable for the conclusions*

## 9. No section carries a heading with nothing under it

A heading over an empty block reads as a page that failed to load, and
it is the first thing a reviewer notices. Sections that have nothing to
say are not rendered; pages are broken so that a section either starts
with enough room to be useful or moves whole to the next page.

- Tested by: `test_the_report_has_something_to_say`,
  `test_no_placeholder_text_survives`, `test_no_page_is_nearly_blank`
- On the page: *No section carries a heading with nothing under it*

---

## Chart conventions

Charts follow IBCS (International Business Communication Standards) so
that a reader who learns the notation once can read every chart in the
pack without consulting a legend.

**Semantic notation.** The appearance of a series is fixed by what the
series *is*, not by where it appears in a list:

| Series | Drawn as |
| --- | --- |
| Actual | Solid fill in the theme accent |
| Plan, budget, target, quota | Outline only, no fill |
| Forecast, projection, estimate | Hatched, at reduced opacity |
| Prior period, last year | Solid neutral grey |

Two solid bars in two accent colours say "two categories". Revenue
against budget is not two categories; it is what happened against what
was committed, and the notation says so before the legend is read. The
series kind is derived from the column name by whole-word match — the
first version matched substrings and classified `monthly_revenue` as a
prior period because `ly` is inside `monthly`.

**Zero baseline.** Bars are read by comparing lengths, so they start at
zero. Four bars between 980 and 1,020 on an axis starting at 975 make
the shortest look a fifth of the tallest; the difference is 4%. The axis
labels that would correct that are the part nobody checks. This is the
one chart rule that changes what a reader concludes rather than how the
chart looks.

**A truncated trend says so.** A line chart may be truncated — sometimes
it must be, or a real movement disappears into the thickness of the line
— but then it carries a scaling indicator naming the floor it was drawn
from.

**Message-first titles.** The title is the finding; the variable names
are the subtitle. *"North is 2.8× the median region"* over
*"revenue by region"*. This is the convention that makes a deck read
faster than a dashboard: the reader takes the point from the title and
uses the axes to check it.

**One theme per domain.** Each domain has its own accent, and the
charts, headings, cover and dashboard tiles all take it from the same
place, so an e-commerce report is not orange with blue bars.

- Built in: `app/engines/chart_engine.py` (interactive),
  `app/engines/chart_exporter.py` (printed)
- Tested by: `backend/tests/test_chart_standards.py`, which measures the
  drawn bars in pixels rather than trusting the plotting library's
  defaults, and `backend/tests/test_chart_theming.py`

---

## What the report must contain

Structure follows the pyramid: the answer first, then what supports it,
then the detail that supports that. A reader who stops after the
executive summary has the conclusion; a reader who continues can check
it.

1. **Executive summary** — the headline finding, its size, and what it
   implies, in the first two sentences.
2. **Findings** — each with its figure, its evidence and its severity.
3. **Dashboard page** — the same tiles as the app, from the same spec,
   so the page and the screen cannot show different views of one file.
4. **Data quality note** — completeness and duplication stated before
   anything is concluded from the data, not after.
5. **Domain section** — the KPIs of the detected domain only. An HR
   report does not carry finance benchmarks.
6. **Recommendations** — each tied to a finding, with the action and
   what it is worth.
7. **Appendix** — analytical method, quality-score formula, reference
   ranges with sources, the review checklist, and the basis of
   preparation.

## Statistical conduct

- Normality is tested, not assumed; the result chooses the downstream
  test.
- Group comparisons check equality of variance before ANOVA and fall
  back to Kruskal-Wallis where it does not hold.
- Families of tests are corrected with Benjamini-Hochberg, and findings
  are filtered on the q-value rather than the raw p. Twenty column pairs
  at p<0.05 produce one false positive by construction.
- Effect sizes below the level that would change a decision are left
  out rather than reported with a caveat.
- Outliers are reported, never silently removed. An extreme value is
  frequently the finding.
- Where a question cannot be answered from the data supplied, the report
  says so rather than answering it from general expectation.

## Sign-off

The checklist above is run against the report being produced — the
prose and the figures the reader is holding, not the intention — and
printed in the appendix as **D. Review Checklist**, with the basis of
each result.

It can fail, which is the point. An exception is printed on the page
with what triggered it. A report that discloses one finding it could not
trace is worth more than one that quietly dropped it, and a checklist
that cannot fail is decoration.

- Built in: `backend/app/engines/report_signoff.py`
- Tested by: `backend/tests/test_signoff.py`, which is mostly about the
  failing side

The checklist deliberately keeps its own copy of the rules rather than
importing them from the engines it checks, as does
`test_report_standard.py`. Three independent statements of the same
standard disagree loudly when one of them drifts; one shared
implementation would pass itself by construction.
