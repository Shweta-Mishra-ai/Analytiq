"""
engines/report_theme.py — the colours of a report, shared by the PDF
builders and the charts printed in them.

Each theme is one dark primary (headings, table headers, cover), one
accent (rules and chart marks) and a pale tint, on shared neutrals. The
charts used their own hard-coded blues whatever the report's colours, and
the themes mixed saturated fills (orange headers, pink rows, brown covers)
that fought each other on the page. Status colours are reserved for status.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Shared by every theme: text, rules and the meaning of status colours.
_NEUTRAL = {
    "text":        "#1F2937", "text_muted":  "#6B7280",
    "border":      "#E5E7EB", "bg_card":     "#F9FAFB",
    "positive":    "#067647", "negative":    "#B42318",
    "warning":     "#B54708", "info":        "#475467",
    "critical_bg": "#FEF3F2", "warning_bg":  "#FFFAEB",
    "positive_bg": "#ECFDF3", "info_bg":     "#F2F4F7",
    "header_text": "#FFFFFF",
}


def _theme(primary: str, accent: str, tint: str, label: str) -> dict:
    return {
        **_NEUTRAL,
        "primary":      primary,
        # Older keys, kept so every drawing helper reads the same palette.
        "cover_bg":     primary, "cover_accent": accent,
        "header_bg":    primary,
        "accent":       accent,  "accent2":      tint,
        "bg_light":     tint,
        "domain_label": label,   "domain_badge": accent,
    }


# One corporate palette for every report: deep navy for headings, table
# headers and the cover band, one mid blue for rules and chart marks, a pale
# blue-grey tint. Consulting deliverables use a single restrained house
# palette whatever the subject; the per-domain themes (orange for
# e-commerce, green for sales, brown covers) read as a template, not a firm.
_NAVY, _BLUE, _TINT = "#0B1F3A", "#1F5BA8", "#EEF2F7"

THEMES = {
    "Corporate":        _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
    # Older names still resolve, to the same palette.
    "Corporate Light":  _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
    "HR Blue":          _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
    "Ecommerce Orange": _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
    "Sales Green":      _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
    "Dark Tech":        _theme(_NAVY, _BLUE, _TINT, "BUSINESS ANALYTICS"),
}

# What the cover says the report is about; the colours do not change.
DOMAIN_LABELS = {
    "hr":        "HR & PEOPLE ANALYTICS",
    "ecommerce": "E-COMMERCE ANALYTICS",
    "sales":     "SALES PERFORMANCE ANALYTICS",
    "finance":   "FINANCIAL ANALYTICS",
    "general":   "BUSINESS ANALYTICS",
}

DOMAIN_THEMES = {d: "Corporate" for d in DOMAIN_LABELS}


def resolve(theme_name: str = "", domain: str = "general") -> str:
    """The theme to use: the one asked for if it exists, else the house one."""
    if theme_name in THEMES:
        return theme_name
    return "Corporate"


def get(theme_name: str = "", domain: str = "general") -> dict:
    T = dict(THEMES[resolve(theme_name, domain)])
    T["domain_label"] = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["general"])
    return T
