"""
engines/pdf/predictive_sections.py — the model-based sections.

Split out of narrative_sections when that module passed 900 lines. The
separation is real rather than cosmetic: everything here depends on a
fitted model, and each piece is skipped entirely when there is no usable
one — which is a different failure mode from a descriptive section, where
the data is either present or it is not.

  _model_note        which model was chosen, and how far to trust it
  _decision_table    what acting on the top N% would actually yield
  _leakage_note      fields excluded for knowing the answer in advance
  _predictive_section the section itself, including the no-signal case
"""
import io
import logging

import numpy as np
import pandas as pd

from reportlab.lib.units import mm
from reportlab.lib.colors import white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
)

from app.engines.pdf.theme import (
    _c, W, H, CW_DEFAULT, FONT_BODY, FONT_BOLD, FONT_ITALIC,
    FONT_SERIF, FONT_SERIF_BOLD,
)
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _clean,
    _exhibit, _exhibit_source,
)

logger = logging.getLogger(__name__)


def _model_note(story, s, T, dr, CW):
    """How the model was chosen, and how far its scores can be trusted.

    A report that names a method without saying what it was measured
    against asks the reader to take it on faith. Naming the alternatives
    and their scores is what makes the choice checkable.
    """
    choice = getattr(dr, "model_choice", None)
    if choice is None:
        return

    story.append(Paragraph("How This Model Was Selected", s["h3"]))
    _exhibit(story, s, T, "Candidate models and their cross-validated "
                          "ranking quality")
    rows = []
    for name, auc in (choice.candidates or []):
        rows.append([name + ("  (selected)" if name == choice.name else ""),
                     "{:.3f}".format(auc)])
    if rows:
        _gtable(story, T, ["Model considered", "Cross-validated AUC"], rows,
                [CW * 0.62, CW * 0.38])
        _exhibit_source(story, s, T,
                        "Five-fold cross-validation on the submitted "
                        "dataset.")
        story.append(Spacer(1, 2 * mm))

    parts = [
        "<b>{}</b> was selected on cross-validated ranking quality, not by "
        "preference — on a different dataset a different one wins.".format(
            choice.name)]
    if choice.calibrated and choice.calibration_before is not None:
        parts.append(
            "Its raw scores were {:.0f} percentage points away from observed "
            "risk in the top band; after calibration they are {:.0f}. That "
            "means a score of 0.30 now genuinely corresponds to roughly a "
            "30% chance, so the number can be quoted and not only ranked."
            .format(choice.calibration_before, choice.calibration_after or 0))
    parts.append(
        "The operating threshold is set at <b>{:.2f}</b> rather than the "
        "conventional 0.50, because it {}. On imbalanced data the default "
        "cut discards recall the model has already earned.".format(
            choice.threshold, choice.threshold_basis))
    _narrative_box(story, s, T, " ".join(parts))
    story.append(Spacer(1, 2 * mm))

    excluded = list(getattr(choice, "excluded_high_cardinality", None) or [])
    if excluded:
        story.append(Paragraph(
            "Excluded from the model for having too many distinct values to "
            "learn from: {}. These may still matter; they need grouping into "
            "fewer categories before a model can use them.".format(
                ", ".join("{} ({:,} values)".format(c, n)
                          for c, n in excluded[:4])), s["note"]))
        story.append(Spacer(1, 2 * mm))


def _decision_table(story, s, T, dr, CW):
    """What acting on the top N% of the ranking would actually yield.

    AUC answers "does the model rank correctly". It does not answer the
    question a manager asks, which is "we can contact 200 people this
    month — which 200, and how many of them were going to leave anyway?"
    Precision, recall and lift at a chosen budget are the form in which a
    model becomes a decision.
    """
    bands = list(getattr(dr, "decision_bands", None) or [])
    if not bands:
        return

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Where to Act", s["h3"]))
    _exhibit(story, s, T, "Expected yield at each size of intervention")
    story.append(Paragraph(
        "Records ranked by predicted risk. Each row is a different size of "
        "intervention: how many records it covers, how many of the events "
        "it would reach, and how much better that is than choosing at "
        "random.", s["body"]))
    story.append(Spacer(1, 2 * mm))

    rows = [["If you act on", "Records", "Events reached",
             "Hit rate", "Share of all events", "vs random"]]
    for b in bands:
        rows.append([
            "top {}%".format(b.budget_pct),
            "{:,}".format(b.n_targeted),
            "{:,} of {:,}".format(b.n_events_caught, b.total_events),
            "{:.0f}%".format(b.precision),
            "{:.0f}%".format(b.recall),
            "{:.1f}x".format(b.lift),
        ])
    _gtable(story, T, rows[0], rows[1:],
            [CW * 0.16, CW * 0.13, CW * 0.22, CW * 0.13, CW * 0.22, CW * 0.14])
    _exhibit_source(story, s, T,
                    "Cross-validated predictions on the submitted dataset.")

    best = max(bands, key=lambda b: b.lift)
    _narrative_box(
        story, s, T,
        "<b>Reading this:</b> targeting the top {}% — {:,} records — reaches "
        "{:,} of the {:,} events in the data. {:.0f}% of those contacted "
        "record the event, against {:.0f}% if the same number were chosen at "
        "random, so the effort goes {:.1f} times further. Which row to choose "
        "is a budget decision, not a modelling one.".format(
            best.budget_pct, best.n_targeted, best.n_events_caught,
            best.total_events, best.precision,
            best.precision / best.lift if best.lift else 0, best.lift))
    story.append(Spacer(1, 2 * mm))

    choice = getattr(dr, "model_choice", None)
    gap = ((choice.calibration_after if choice is not None
            and choice.calibration_after is not None
            else getattr(dr, "calibration_gap", None)))
    if gap is not None and gap > 10:
        # The rates above come from observed outcomes, so they stand. The
        # model's own probability scores do not, and someone will
        # eventually read one as "this record has a 70% chance".
        story.append(Paragraph(
            "The rates above are what actually happened in each band, so "
            "they can be relied on. The model's individual risk scores are "
            "not calibrated to probabilities ({:.0f} percentage points out "
            "in the top band), so use the ranking to prioritise and not the "
            "score as a likelihood for any one record.".format(gap),
            s["note"]))
        story.append(Spacer(1, 2 * mm))


def _leakage_note(story, s, T, dr):
    """Name any column that predicts the outcome suspiciously well.

    A field populated only once the outcome is known makes a model look
    excellent in validation and useless in production. Worth saying in the
    report, because the fix is upstream in how the data is recorded.
    """
    findings = list(getattr(dr, "leakage", None) or [])
    if not findings:
        return
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Fields Excluded as Outcome Leakage", s["h3"]))
    for f in findings[:4]:
        _narrative_box(story, s, T, "<b>{}</b> — {}".format(
            getattr(f, "column", "?"), getattr(f, "reason", "")))
        story.append(Spacer(1, 2 * mm))


def _predictive_section(story, s, T, dr, CW, avg_salary_k: float = 0.0,
                        top_cluster=None, driver_chart=None,
                        risk_heatmap=None):
    """Model drivers, honest accuracy, and the highest-risk segment.

    `dr` is a predictive.DriverResult, or None to skip the section. This
    engine has been in the codebase throughout; until now build_pdf had no
    parameter to receive its output, so it ran and was discarded.
    """
    if dr is None:
        return

    tgt = str(dr.target).replace("_", " ").title()

    verdict = getattr(dr, "verdict", None)
    if verdict is not None and not verdict.usable:
        # Say that the data does not support a prediction. Dropping the
        # section silently is indistinguishable from never having tried,
        # and "we looked and found nothing" is a real result a client
        # should be told — particularly before they act as though there
        # were a signal.
        _sec(story, s, T, "Predictive Risk Analysis",
             "Whether {} can be predicted from the rest of the "
             "dataset".format(tgt))
        story.append(Spacer(1, 3 * mm))
        _narrative_box(story, s, T,
                       "<b>No predictive signal found.</b> " + verdict.verdict)
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "This is a finding, not a gap in the analysis: on this data, "
            "{} cannot be anticipated from the fields available. Collecting "
            "the factors thought to drive it — or recording them earlier in "
            "the process — is the prerequisite for a usable model."
            .format(tgt.lower()), s["body"]))
        _leakage_note(story, s, T, dr)
        return

    if not dr.top_drivers:
        return
    _sec(story, s, T, "Predictive Risk Analysis",
         "A model trained to predict {} — drivers, accuracy, and the "
         "highest-risk segment".format(tgt))
    story.append(Spacer(1, 3 * mm))

    # NaN-safe: an undertrained model returns NaN rather than a number, and
    # "nan" must never reach the page.
    auc_txt = "—" if (dr.auc != dr.auc) else "{:.2f}".format(dr.auc)
    acc_txt = ("—" if (dr.accuracy != dr.accuracy)
               else "{:.0f}%".format(dr.accuracy * 100))
    auc_quality = (("strong" if dr.auc >= 0.8 else
                    "moderate" if dr.auc >= 0.7 else "weak")
                   if dr.auc == dr.auc else "not available")
    _kpi_row(story, s, T, [
        {"label": "MODEL AUC", "value": auc_txt,
         "sub": "{} separation".format(auc_quality), "color": T["accent"]},
        {"label": "ACCURACY", "value": acc_txt, "sub": "cross-validated",
         "color": T["positive"]},
        {"label": "RECORDS USED", "value": "{:,}".format(dr.n_rows),
         "sub": "{} features".format(dr.n_features), "color": T["accent"]},
        {"label": "BASE RATE", "value": "{:.0f}%".format(dr.base_rate),
         "sub": "overall event rate", "color": T["text_muted"]},
    ], CW)
    story.append(Spacer(1, 3 * mm))

    _narrative_box(
        story, s, T,
        "<b>How to read this:</b> the model was validated on held-out data, "
        "so the {} AUC reflects genuine predictive power rather than "
        "memorisation — 0.5 is a coin flip, 1.0 is perfect. The importances "
        "below show what the model relies on most. They are predictive, not "
        "proven causes.".format(auc_txt))
    story.append(Spacer(1, 2 * mm))

    _model_note(story, s, T, dr, CW)

    story.append(Paragraph("Top Predictive Drivers", s["h3"]))
    embedded = False
    if driver_chart:
        try:
            story.append(Image(io.BytesIO(driver_chart),
                               width=CW, height=CW * 0.42))
            story.append(Spacer(1, 3 * mm))
            embedded = True
        except Exception:
            logger.warning("driver chart embed failed", exc_info=True)
    if not embedded:
        top_imp = dr.top_drivers[0][1] or 1.0
        rows = [[Paragraph("<b>#</b>", s["sm"]),
                 Paragraph("<b>Driver</b>", s["sm"]),
                 Paragraph("<b>Predictive weight</b>", s["sm"]),
                 Paragraph("<b>Relative importance</b>", s["sm"])]]
        for i, (col, imp) in enumerate(dr.top_drivers, 1):
            bar_w = max(1, int(round(imp / top_imp * 28)))
            rows.append([
                Paragraph(str(i), s["sm"]),
                Paragraph(str(col).replace("_", " "), s["sm"]),
                Paragraph("{:.1f}%".format(imp), s["sm"]),
                Paragraph('<font color="{}">{}</font>'.format(
                    # U+2588 full block is absent from Carlito; U+25AA
                    # is present and reads as a bar at this size.
                    T["accent"], "\u25AA" * bar_w), s["sm"]),
            ])
        tbl = Table(rows, colWidths=[CW * 0.06, CW * 0.40, CW * 0.18, CW * 0.36])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  _c(T["header_bg"])),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _c(T["bg_light"])]),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("GRID",           (0, 0), (-1, -1), 0.3, _c(T["border"])),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 3 * mm))

    if top_cluster is not None and getattr(top_cluster, "n_events", 0) >= 10:
        tc = top_cluster
        lift = tc.rate / tc.base_rate if tc.base_rate else 0
        _narrative_box(
            story, s, T,
            "<b>Largest risk cluster:</b> records where <b>{}</b> — {:,} of "
            "them — show a {:.0f}% event rate ({:.1f}x the {:.0f}% base) and "
            "account for <b>{:.0f}% of all events</b> in the dataset. This is "
            "the most concentrated addressable pocket of risk: a targeted "
            "intervention here reaches the most affected records for the "
            "least effort.".format(
                tc.description, tc.n, tc.rate, lift, tc.base_rate,
                tc.share_of_events))
        story.append(Spacer(1, 3 * mm))

    if risk_heatmap:
        try:
            story.append(KeepTogether([
                Paragraph("Risk Concentration Map", s["h3"]),
                Image(io.BytesIO(risk_heatmap), width=CW * 0.90,
                      height=CW * 0.60),
                Paragraph("Darker cells carry a higher event rate. The "
                          "hottest cell is the segment to address first.",
                          s["sm"]),
            ]))
            story.append(Spacer(1, 3 * mm))
        except Exception:
            logger.warning("risk heatmap embed failed", exc_info=True)

    if dr.high_risk_n >= 10 and dr.high_risk_rate > 0:
        lift = dr.high_risk_rate / dr.base_rate if dr.base_rate else 0
        prof = dr.high_risk_profile or "the model's highest-probability profile"
        _narrative_box(
            story, s, T,
            "<b>Highest-risk segment:</b> {:,} records fall in the model's "
            "top-risk quintile and show an actual event rate of "
            "<b>{:.0f}%</b> — {:.1f}x the {:.0f}% base rate. Shared profile: "
            "{}. This is where intervention has the highest expected return; "
            "pull this list from the source system and act on it "
            "first.".format(dr.high_risk_n, dr.high_risk_rate, lift,
                            dr.base_rate, prof))
        story.append(Spacer(1, 3 * mm))

        expected_events = int(round(dr.high_risk_n * dr.high_risk_rate / 100.0))
        avoidable = int(round(dr.high_risk_n *
                              max(dr.high_risk_rate - dr.base_rate, 0) / 100.0))
        story.append(Paragraph("Scenario and Expected Value", s["h3"]))
        if avg_salary_k and avg_salary_k > 0 and avoidable > 0:
            lo = avoidable * avg_salary_k * 0.5
            hi = avoidable * avg_salary_k * 2.0
            roi_line = (
                " Costed at the {:,.0f}K replacement cost supplied for this "
                "report and the published 50-200% band, the avoidable share "
                "is roughly <b>{:,.0f}K-{:,.0f}K</b> per cycle. That unit "
                "cost is an assumption you supplied, not a figure measured "
                "from this data — substitute your own for a board-ready "
                "number.".format(
                    avg_salary_k, lo, hi))
        else:
            roi_line = (
                " No replacement cost was supplied with this report, so no "
                "monetary figure is asserted here. Enter one at report setup "
                "to translate the avoidable events into a range.")
        _narrative_box(
            story, s, T,
            "<b>If nothing changes:</b> at the segment's current rate, about "
            "<b>{:,}</b> of these {:,} records are expected to record the "
            "event next cycle. <b>Roughly {:,}</b> of those are potentially "
            "avoidable — the excess above the {:.0f}% base rate — if the "
            "drivers above are addressed for this segment.{}".format(
                expected_events, dr.high_risk_n, avoidable, dr.base_rate,
                roi_line))

    _decision_table(story, s, T, dr, CW)
    _leakage_note(story, s, T, dr)
