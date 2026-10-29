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


THEMES = {
    "Corporate Light":  _theme("#13294B", "#2E5AAC", "#EEF2FA", "BUSINESS ANALYTICS"),
    "HR Blue":          _theme("#102A43", "#2F6690", "#EDF3F8", "HR & PEOPLE ANALYTICS"),
    "Ecommerce Orange": _theme("#3A2414", "#C05621", "#FDF3EC", "E-COMMERCE ANALYTICS"),
    "Sales Green":      _theme("#123524", "#2F7D59", "#EEF6F1", "SALES PERFORMANCE ANALYTICS"),
    # Reports are printed on white. This theme used light text for a dark
    # screen, which came out near-invisible on the page.
    "Dark Tech":        _theme("#0D1117", "#2F81F7", "#EEF4FF", "TECHNICAL ANALYTICS"),
}

DOMAIN_THEMES = {
    "hr":        "HR Blue",
    "ecommerce": "Ecommerce Orange",
    "sales":     "Sales Green",
    "finance":   "Corporate Light",
    "general":   "Corporate Light",
}


def resolve(theme_name: str = "", domain: str = "general") -> str:
    """The theme to use: the one asked for if it exists, else the domain's."""
    if theme_name in THEMES:
        return theme_name
    return DOMAIN_THEMES.get(domain, "Corporate Light")


def get(theme_name: str = "", domain: str = "general") -> dict:
    return THEMES[resolve(theme_name, domain)]
