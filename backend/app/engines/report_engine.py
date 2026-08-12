import logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportTheme:
    name: str
    page_bg:           tuple = (255, 255, 255)
    page_margin_mm:    int   = 18
    accent:            tuple = (26,  74,  138)
    accent_light:      tuple = (235, 242, 255)
    text_dark:         tuple = (30,  30,   40)
    text_muted:        tuple = (100, 105, 130)
    text_light:        tuple = (255, 255, 255)
    header_bg:         tuple = (26,  74,  138)
    header_text:       tuple = (255, 255, 255)
    table_header_bg:   tuple = (26,  74,  138)
    table_header_text: tuple = (255, 255, 255)
    table_row_alt:     tuple = (245, 248, 255)
    table_row_main:    tuple = (255, 255, 255)
    table_border:      tuple = (210, 220, 240)
    chart_colors:      list  = field(default_factory=lambda: [
        "#1a4a8a", "#2196F3", "#42A5F5",
        "#90CAF9", "#0D47A1", "#1565C0"
    ])
    positive_color:    tuple = (34,  139,  84)
    negative_color:    tuple = (200,  50,  50)
    neutral_color:     tuple = (26,   74, 138)


THEMES = {
    "Corporate Light": ReportTheme(
        name               = "Corporate Light",
        page_bg            = (255, 255, 255),
        accent             = (26,   74, 138),
        accent_light       = (235, 242, 255),
        text_dark          = (30,   30,  40),
        text_muted         = (100, 105, 130),
        text_light         = (255, 255, 255),
        header_bg          = (26,   74, 138),
        header_text        = (255, 255, 255),
        table_header_bg    = (26,   74, 138),
        table_header_text  = (255, 255, 255),
        table_row_alt      = (245, 248, 255),
        table_row_main     = (255, 255, 255),
        table_border       = (210, 220, 240),
        chart_colors       = ["#1a4a8a", "#2196F3", "#42A5F5",
                               "#90CAF9", "#0D47A1", "#1565C0"],
        positive_color     = (34,  139,  84),
        negative_color     = (200,  50,  50),
        neutral_color      = (26,   74, 138),
    ),
    "Dark Tech": ReportTheme(
        name               = "Dark Tech",
        page_bg            = (7,    8,   15),
        accent             = (79,  142, 247),
        accent_light       = (20,   25,  50),
        text_dark          = (221, 225, 245),
        text_muted         = (100, 105, 140),
        text_light         = (221, 225, 245),
        header_bg          = (14,   15,  26),
        header_text        = (221, 225, 245),
        table_header_bg    = (79,  142, 247),
        table_header_text  = (255, 255, 255),
        table_row_alt      = (14,   15,  26),
        table_row_main     = (10,   11,  20),
        table_border       = (30,   32,  53),
        chart_colors       = ["#4f8ef7", "#22d3a5", "#f7934f",
                               "#a78bfa", "#f77070", "#ffd43b"],
        positive_color     = (34,  211, 165),
        negative_color     = (247, 112, 112),
        neutral_color      = (79,  142, 247),
    ),
    "Executive Green": ReportTheme(
        name               = "Executive Green",
        page_bg            = (255, 255, 255),
        accent             = (26,  107,  74),
        accent_light       = (235, 248, 242),
        text_dark          = (20,   40,  30),
        text_muted         = (90,  115, 100),
        text_light         = (255, 255, 255),
        header_bg          = (26,  107,  74),
        header_text        = (255, 255, 255),
        table_header_bg    = (26,  107,  74),
        table_header_text  = (255, 255, 255),
        table_row_alt      = (240, 250, 245),
        table_row_main     = (255, 255, 255),
        table_border       = (200, 230, 215),
        chart_colors       = ["#1a6b4a", "#2ecc71", "#27ae60",
                               "#82e0aa", "#145a32", "#1e8449"],
        positive_color     = (26,  107,  74),
        negative_color     = (192,  57,  43),
        neutral_color      = (26,  107,  74),
    ),
}


@dataclass
class ReportConfig:
    title:        str  = "Data Analysis Report"
    client_name:  str  = "Client"
    prepared_by:  str  = "Analytiq"
    confidential: bool = True
    theme_name:   str  = "Corporate Light"

    @property
    def theme(self) -> ReportTheme:
        return THEMES.get(self.theme_name, THEMES["Corporate Light"])


@dataclass
class ReportSection:
    title:       str
    content:     str
    chart_image: Optional[bytes]        = None
    table_data:  Optional[pd.DataFrame] = None


@dataclass
class ReportData:
    config:       ReportConfig
    df:           pd.DataFrame
    sections:     List[ReportSection]
    generated_at: str = ""

    def __post_init__(self):
        self.generated_at = datetime.now().strftime("%B %d, %Y")
