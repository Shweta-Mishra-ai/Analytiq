import logging
import pandas as pd
from app.config import config

logger = logging.getLogger(__name__)


def build_chat_system_prompt(df: pd.DataFrame) -> str:
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    sample = (
        df.head(config.max_rows_llm_context)
          .iloc[:, :config.max_cols_llm_context]
          .to_string(index=False, max_rows=10)
    )

    return f"""You are Analytiq — a data analyst assistant.

DATASET:
  Rows    : {len(df):,}
  Columns : {len(df.columns)}
  Numeric : {num_cols}
  Category: {cat_cols}
  Dates   : {date_cols}

SAMPLE DATA:
{sample}

Return ONLY a valid JSON object. No text before. No text after. No markdown.

SCHEMA:
{{"tool": "<name>", "params": {{...}}, "explanation": "<plain English>"}}

AVAILABLE TOOLS:
  aggregate       → group_col, value_col, agg_func (sum|mean|count|max|min)
  filter          → column, operator (>|<|==|!=|contains), value
  top_n           → sort_col, n, ascending (true|false)
  describe_column → column
  correlation     → col_a, col_b
  count_values    → column
  plot_bar        → x, y, title
  plot_line       → x, y, title
  plot_scatter    → x, y, color (optional), title
  plot_histogram  → column, nbins, title
  plot_pie        → names_col, values_col, title
  plot_heatmap    → (no params needed)
  none            → (when nothing matches)

RULES:
1. Only use column names that exist in DATASET above.
2. explanation must be plain English — no code, no jargon.
3. If nothing matches return:
   {{"tool": "none", "params": {{}}, "explanation": "I could not find a matching analysis."}}
"""


def get_df_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    lines = [
        f"Rows: {len(df):,}",
        f"Columns: {list(df.columns)}",
        f"Missing: {df.isnull().sum().sum():,}",
    ]
    if num_cols:
        lines.append(df[num_cols].describe().round(2).to_string())
    return "\n".join(lines)
