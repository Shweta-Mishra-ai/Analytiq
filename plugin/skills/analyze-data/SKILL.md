---
description: Analyses a CSV, Excel or JSON file with the Analytiq tools and, when asked, writes a client-ready PDF report. Use when the user asks to analyse, summarise or report on a spreadsheet or tabular data file.
---

Use the Analytiq MCP tools for this; do not read the file's rows yourself.

1. `load_dataset` with the file's path. It returns the dataset id and a profile.
2. `business_summary` for the headline figures, key messages and tested findings.
3. `query` for specific questions (totals by group, filters, top rows, correlations).
4. `find_drivers` or `train_model` when the user asks what predicts an outcome.
5. `generate_report` when the user wants a PDF (`language` "en" or "hi").

Report the figures exactly as the tools return them. Where a tool says a difference is within chance, say so rather than ranking the groups.
