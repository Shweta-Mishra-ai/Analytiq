"""
app/mcp_server.py — Analytiq as tools Claude can call (Model Context Protocol).

Claude reading a spreadsheet row by row spends tokens on every cell and does
the arithmetic in its head. Through these tools the analysis runs here, in
the same tested engines as the web app, and Claude receives a few hundred
tokens of results: the scorecard, the tested findings, a model's verdict, the
path or link of a finished PDF. The size of the file no longer sets the cost.

One set of tools, two ways to reach it:

* Local (stdio) — `python -m app.mcp_server`, launched by Claude Desktop or
  Claude Code on the user's own machine. Files are read from disk by path and
  nothing leaves the machine.
* Remote (Streamable HTTP) — mounted at `/mcp` inside the web app, for
  claude.ai and for every member of a team. Each caller sends their own
  Analytiq token (`Authorization: Bearer …`, entered once as the connector's
  request header) and sees only their own datasets. Local paths are refused
  there; data comes from datasets uploaded to the app, an https link, or CSV
  text.

Every tool answers in short Markdown, never raw rows: a tool that returned
the data would save nothing.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import ipaddress
import logging
import os
import secrets
import socket
import time
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

logger = logging.getLogger(__name__)

# Text pasted into a tool call travels as tokens; beyond this a file belongs
# in the web app or behind a link.
MAX_CSV_TEXT = 200_000
MAX_URL_BYTES = 50 * 1024 * 1024
MAX_TABLE_ROWS = 15
DOWNLOAD_TTL = 3600

INSTRUCTIONS = """\
Analytiq runs statistical and business analysis on tabular data (CSV, Excel,
JSON) and writes consulting-style PDF reports in English or Hindi.

Typical order: load_dataset (or list_datasets), then business_summary, then
query for specific questions and generate_report for a PDF. Figures in the
responses are computed by Analytiq from the data; group differences and
correlations are tested for chance and adjusted for multiple comparisons."""


def _read(title: str) -> ToolAnnotations:
    return ToolAnnotations(title=title, read_only_hint=True, destructive_hint=False,
                           idempotent_hint=True, open_world_hint=False)


def _write(title: str, open_world: bool = False) -> ToolAnnotations:
    """Creates something new (a dataset, a model, a PDF); never deletes or
    overwrites what is there."""
    return ToolAnnotations(title=title, read_only_hint=False, destructive_hint=False,
                           idempotent_hint=False, open_world_hint=open_world)


mcp = MCPServer(name="Analytiq", instructions=INSTRUCTIONS)


# ── who is calling ───────────────────────────────────────────────────────

class AccessError(Exception):
    pass


def _owner(ctx: Optional[Context]) -> tuple[str, bool]:
    """(owner, is_local). No HTTP headers means stdio: the user's own
    machine, the same owner as the web app in single-user mode. Over HTTP
    the bearer token is verified, not trusted."""
    from app.services.auth import LOCAL_OWNER, _open_mode, _user_for_token
    headers = None
    try:
        headers = ctx.headers if ctx is not None else None
    except Exception:
        headers = None
    if headers is None:
        return LOCAL_OWNER, True
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    user = _user_for_token(token) if token else None
    if user is None and token:
        # The key from the account page: made for a connector, which has
        # no way to sign in again when a 30-day token runs out.
        from app.services.user_store import user_store
        user = user_store.user_for_connector_key(token)
    if user is not None:
        return user.username, False
    if _open_mode() and headers.get("x-analytiq-loopback") == "1":
        return LOCAL_OWNER, False
    raise AccessError("Not signed in: this connector's key is missing, replaced or "
                      "removed. Create a key on the Account page of the Analytiq web "
                      "app and set the connector's request header to "
                      "'Authorization: Bearer <key>'.")


def _store():
    from app.services.dataset_store import store
    return store


def _df(owner: str, dataset_id: str) -> pd.DataFrame:
    df = _store().get_df(owner, dataset_id)
    if df is None:
        raise ValueError(f"No dataset '{dataset_id}'. Call list_datasets to see yours.")
    return df


def _guard(fn):
    """Turn expected failures into a plain answer the model can relay."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (AccessError, ValueError) as e:
            return f"**Could not do that:** {e}"
        except Exception as e:
            # The busy and over-quota answers of the web API, in words.
            detail = getattr(e, "detail", None)
            if isinstance(detail, str) and getattr(e, "status_code", 0) in (413, 503, 507):
                return f"**Could not do that:** {detail}"
            logger.exception("MCP tool %s failed", fn.__name__)
            return f"**The analysis failed:** {e}"
    return wrapper


# ── formatting ───────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, limit: int = MAX_TABLE_ROWS) -> str:
    from app.engines.chart_exporter import fmt
    if df is None or df.empty:
        return ""
    shown = df.head(limit)

    def cell(v):
        if isinstance(v, float):
            return fmt(v)
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        return str(v).replace("|", "\\|")[:40]
    head = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in shown.itertuples(index=False)]
    more = f"\n_{len(df) - limit:,} more rows not shown._" if len(df) > limit else ""
    return "\n".join([head, rule, *rows]) + more


def _profile(meta, df: pd.DataFrame) -> str:
    from app.engines.domains.base import is_id_column
    from app.services.dtypes import text_columns
    ids = [c for c in df.columns if is_id_column(c, df[c])]
    nums = [c for c in df.select_dtypes(include="number").columns if c not in ids]
    dates = df.select_dtypes(include="datetime").columns.tolist()
    texts = [c for c in text_columns(df) if c not in ids]
    missing = float(df.isna().mean().mean() * 100) if df.size else 0.0
    lines = [f"**Dataset `{meta.dataset_id}`** — {meta.filename}: "
             f"{len(df):,} rows × {df.shape[1]} columns, {missing:.1f}% of cells blank.",
             f"- Numbers: {', '.join(nums[:12]) or '—'}",
             f"- Categories/text: {', '.join(texts[:12]) or '—'}",
             f"- Dates: {', '.join(dates) or '—'}",
             f"- Identifiers (not analysed as measures): {', '.join(ids) or '—'}"]
    if meta.warnings:
        lines.append("- Notes: " + "; ".join(str(w) for w in meta.warnings[:4]))
    lines.append("\nNext: `business_summary` for the headline figures and findings.")
    return "\n".join(lines)


# ── loading ──────────────────────────────────────────────────────────────

def _create(owner: str, name: str, data: bytes):
    from app.api.datasets import _UploadShim
    from app.engines.data_loader import load_file
    from app.engines.data_validator import validate_dataframe, validate_file_size
    ok, msg = validate_file_size(len(data))
    if not ok:
        raise ValueError(msg)
    from app.services.upload_guard import UploadRejected, check
    try:
        check(name, data)
    except UploadRejected as e:
        raise ValueError(str(e))
    from fastapi import HTTPException
    from app.services.limits import check_storage
    try:
        check_storage(owner, len(data))
    except HTTPException as e:
        raise ValueError(str(e.detail))
    result = load_file(_UploadShim(name, data))
    if not result.success or result.df is None:
        raise ValueError(result.error or "The file could not be read.")
    validation = validate_dataframe(result.df)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))
    meta = _store().create(owner, result.df, name, size_mb=len(data) / 2**20,
                           sheet_names=getattr(result, "sheet_names", None) or [],
                           warnings=(result.warnings or []) + (validation.warnings or []))
    return meta, _store().get_df(owner, meta.dataset_id)


def _public_host(host: str) -> bool:
    """Whether every address `host` resolves to is public. A link to
    localhost, the cloud metadata address or a private network would let
    a caller make the server fetch what only it can reach."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):          # a redirect could point inward
        return None


def _fetch(url: str) -> tuple[str, bytes]:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("Only https links can be loaded.")
    if not _public_host(parts.hostname):
        raise ValueError("That link points to a private or local address.")
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "Analytiq-MCP"})
    with opener.open(req, timeout=30) as resp:
        data = resp.read(MAX_URL_BYTES + 1)
    if len(data) > MAX_URL_BYTES:
        raise ValueError("The file at that link is larger than 50 MB.")
    name = os.path.basename(parts.path) or "download.csv"
    return name, data


# ── tools ────────────────────────────────────────────────────────────────

@mcp.tool(annotations=_read("List datasets"))
@_guard
def list_datasets(ctx: Context) -> str:
    """List the datasets you have loaded, newest first, with their ids."""
    owner, _ = _owner(ctx)
    metas = sorted(_store().list_meta(owner), key=lambda m: -m.uploaded_at)
    if not metas:
        return "No datasets yet. Use `load_dataset` with a file path, an https link or CSV text."
    return "\n".join(f"- `{m.dataset_id}` — {m.filename} ({m.rows:,} rows × {m.cols} columns)"
                     for m in metas[:30])


@mcp.tool(annotations=_write("Load a dataset", open_world=True))
@_guard
def load_dataset(ctx: Context, path: str = "", url: str = "", csv_text: str = "",
                 name: str = "") -> str:
    """Load a table (CSV, Excel, JSON) for analysis and get its profile and id.

    Give exactly one of: `path` (a file on this computer; local use only),
    `url` (an https link to the file) or `csv_text` (CSV content, up to
    200 KB). `name` labels the dataset when the source has no file name.
    """
    owner, local = _owner(ctx)
    given = [x for x in (path, url, csv_text) if x]
    if len(given) != 1:
        raise ValueError("Give exactly one of path, url or csv_text.")
    if path:
        if not local:
            raise ValueError("File paths work only when Analytiq runs on your own "
                             "computer. Upload the file in the web app, or pass an "
                             "https link or CSV text.")
        full = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(full):
            raise ValueError(f"No file at {full}.")
        with open(full, "rb") as f:
            data = f.read()
        fname = os.path.basename(full)
    elif url:
        fname, data = _fetch(url)
    else:
        if len(csv_text) > MAX_CSV_TEXT:
            raise ValueError("CSV text is limited to 200 KB; use a path, a link or the web app.")
        data, fname = csv_text.encode("utf-8"), (name or "pasted") + ".csv"
    if name and not fname.lower().startswith(name.lower()):
        fname = name + os.path.splitext(fname)[1]
    meta, df = _create(owner, fname, data)
    return _profile(meta, df)


@mcp.tool(annotations=_read("Business summary"))
@_guard
def business_summary(ctx: Context, dataset_id: str) -> str:
    """Headline figures (this period against the last), the key messages
    with their evidence, and the tested findings and risks. Start here."""
    from app.engines.scorecard import build_scorecard, fmt_change, fmt_value
    from app.engines.story_engine import generate_story
    owner, _ = _owner(ctx)
    df = _df(owner, dataset_id)
    out = []
    sc = build_scorecard(df)
    if sc.metrics:
        if sc.has_comparison:
            out.append(f"### Scorecard — {sc.period} vs {sc.prior_period}")
            out.append("| Measure | Now | Prior | Change |\n|---|---|---|---|")
            for m in sc.metrics:
                prior = ("—" if m.prior is None else
                         "{:.1f}%".format(m.prior * 100) if m.kind == "pct" else
                         fmt_value(type(m)(m.label, m.kind, m.prior)))
                out.append(f"| {m.label} | {fmt_value(m)} | {prior} | {fmt_change(m)} |")
        else:
            out.append("### Scorecard")
            out += [f"- {m.label}: {fmt_value(m)}" for m in sc.metrics]
    if sc.messages:
        out.append("\n### Key messages")
        for i, m in enumerate(sc.messages[:4], 1):
            out.append(f"{i}. **{m.title}.** {m.detail} _Action:_ {m.action}")
    story = generate_story(df)
    if not sc.messages and story.executive_summary:
        out.append("### Summary\n" + story.executive_summary)
    if story.key_findings:
        out.append("\n### Findings")
        out += [f"- {f}" for f in story.key_findings[:5]]
    if story.business_risks:
        out.append("\n### Risks")
        out += [f"- {r}" for r in story.business_risks[:4]]
    if story.recommended_actions:
        out.append("\n### Next steps")
        out += [f"- {a}" for a in story.recommended_actions[:4]]
    return "\n".join(out) or "Nothing to report for this dataset."


@mcp.tool(annotations=_read("Data quality"))
@_guard
def data_quality(ctx: Context, dataset_id: str) -> str:
    """Quality score and grade, missing values, duplicates and outliers, and
    whether the data is ready for modelling."""
    from app.engines.health_engine import compute_health
    owner, _ = _owner(ctx)
    df = _df(owner, dataset_id)
    h = compute_health(df)
    lines = [f"**Quality score {h['score']}/100 (grade {h['grade']}, {h['label']}).**",
             f"- Blank cells: {h['missing_pct']}% · duplicate rows: {h['dup_pct']}% · "
             f"columns with outliers: {h['outlier_pct']}%"]
    try:
        from app.engines.readiness import assess_readiness
        r = assess_readiness(df)
        if r.summary:
            lines.append(f"- {r.summary}")
        for issue in r.blockers[:3]:
            lines.append(f"- Blocks modelling — '{issue.column}': {issue.issue} Fix: {issue.fix}")
    except Exception:
        logger.warning("readiness check failed in MCP", exc_info=True)
    worst = df.isna().mean().sort_values(ascending=False)
    worst = worst[worst > 0].head(5)
    if len(worst):
        lines.append("- Most blank: " + ", ".join(f"{c} ({v * 100:.0f}%)" for c, v in worst.items()))
    return "\n".join(lines)


_QUERY_OPS = {"aggregate", "filter", "top_n", "describe_column", "correlation", "count_values"}


@mcp.tool(annotations=_read("Query the data"))
@_guard
def query(ctx: Context, dataset_id: str, operation: str, column: str = "",
          group_by: str = "", agg: str = "sum", operator: str = "", value: str = "",
          other_column: str = "", n: int = 10, ascending: bool = False) -> str:
    """Answer a specific question from the data.

    operation:
    - aggregate: `agg` (sum, mean, median, count, min, max) of `column` by `group_by`
    - filter: rows where `column` `operator` (>, <, ==, !=, contains) `value`
    - top_n: the `n` rows with the highest (or `ascending` lowest) `column`
    - describe_column: typical value, spread and range of `column`
    - correlation: whether `column` and `other_column` move together (tested)
    - count_values: how often each value of `column` occurs
    Rankings and correlations say when a gap is within chance.
    """
    from app.ai.tool_dispatcher import dispatch
    owner, _ = _owner(ctx)
    df = _df(owner, dataset_id)
    if operation not in _QUERY_OPS:
        raise ValueError(f"operation must be one of: {', '.join(sorted(_QUERY_OPS))}")
    params = {
        "aggregate": {"group_col": group_by, "value_col": column, "agg_func": agg},
        "filter": {"column": column, "operator": operator, "value": value},
        "top_n": {"sort_col": column, "n": max(1, min(int(n), 50)), "ascending": ascending},
        "describe_column": {"column": column},
        "correlation": {"col_a": column, "col_b": other_column},
        "count_values": {"column": column},
    }[operation]
    r = dispatch(df, operation, params)
    if not r.success:
        raise ValueError(r.error)
    text = r.text_output or ""
    if operation == "aggregate":
        return text                                 # already lists the groups
    if r.dataframe is not None and operation in ("filter", "top_n", "count_values"):
        text += "\n\n" + _md_table(r.dataframe)
    return text


@mcp.tool(annotations=_read("Find what predicts an outcome"))
@_guard
def find_drivers(ctx: Context, dataset_id: str, outcome: str = "") -> str:
    """Which columns predict a yes/no outcome (churn, attrition, default),
    and whether the prediction beats chance at all. `outcome` defaults to the
    likeliest two-value column."""
    from app.engines.predictive import binary_target_candidates, compute_drivers
    owner, _ = _owner(ctx)
    df = _df(owner, dataset_id)
    target = outcome or next(iter(binary_target_candidates(df)), None)
    if not target:
        raise ValueError("No column has exactly two outcomes (such as yes / no) to predict.")
    if target not in df.columns:
        raise ValueError(f"No column '{target}'.")
    res = compute_drivers(df, target)
    if res is None:
        raise ValueError(f"'{target}' could not be modelled: it needs two classes and enough rows.")
    lines = [f"**Outcome:** {target} = '{res.event_label}' "
             f"(base rate {res.base_rate:.1f}%, {res.n_rows:,} rows)."]
    if res.verdict:
        lines.append(res.verdict)
    if res.signal in ("useful", "weak") and res.top_drivers:
        lines.append("\n| Factor | Share of what the model used |\n|---|---|")
        lines += [f"| {f} | {imp:.0f}% |" for f, imp in res.top_drivers[:8]]
        if res.high_risk_profile:
            lines.append(f"\nHighest-risk group: {res.high_risk_profile} — "
                         f"{res.high_risk_rate:.1f}% against {res.base_rate:.1f}% overall "
                         f"({res.high_risk_n:,} rows).")
    return "\n".join(lines)


@mcp.tool(annotations=_write("Train prediction models"))
@_guard
def train_model(ctx: Context, dataset_id: str, target: str) -> str:
    """Train and compare prediction models for `target`, judged on rows the
    models did not see, against the naive baseline."""
    from app.engines.ml_engine import TargetError, run_ml_pipeline
    owner, _ = _owner(ctx)
    df = _df(owner, dataset_id)
    if target not in df.columns:
        raise ValueError(f"No column '{target}'.")
    from app.services.limits import heavy_job
    try:
        with heavy_job("model training"):
            rep = run_ml_pipeline(df, target)
    except TargetError as e:
        raise ValueError(str(e))
    _store().cache_set(owner, dataset_id, f"ml_{target}", rep)
    _store().cache_set(owner, dataset_id, "ml_last", rep)
    lines = [f"**{rep.summary}**"] if getattr(rep, "summary", "") else []
    if getattr(rep, "no_signal", ""):
        lines.append(rep.no_signal)
    lines.append("\n| Model | Held-out score | Note |\n|---|---|---|")
    for m in rep.models:
        note = ("failed: " + m.error[:60]) if m.error else ("best" if m.is_best else "")
        score = "—" if m.error else f"{m.test_score:.3f} {m.metric_name}"
        lines.append(f"| {m.name} | {score} | {note} |")
    fi = getattr(rep, "feature_importance", None) or []
    if fi and not getattr(rep, "no_signal", ""):
        lines.append("\nTop factors: " + ", ".join(
            str(getattr(f, "feature", None) or (f.get("feature") if isinstance(f, dict) else f))
            for f in fi[:5]))
    for w in (getattr(rep, "warnings", None) or [])[:3]:
        lines.append(f"- ⚠ {w}")
    return "\n".join(lines)


# ── reports ──────────────────────────────────────────────────────────────

def _download_secret() -> bytes:
    from app.services.tokens import _get_secret
    return hashlib.sha256(b"analytiq-report-download|" + _get_secret()).digest()


def _sign(name: str, exp: int) -> str:
    return hmac.new(_download_secret(), f"{name}|{exp}".encode(), "sha256").hexdigest()[:32]


def report_dir() -> str:
    from app.config import config
    d = os.path.join(config.data_dir, "mcp_reports")
    os.makedirs(d, exist_ok=True)
    return d


def sweep_reports(now: Optional[float] = None) -> int:
    """Delete report files whose download links have expired. They used to
    stay on disk forever after the one-hour link stopped working."""
    now = time.time() if now is None else now
    removed = 0
    d = report_dir()
    for name in os.listdir(d):
        path = os.path.join(d, name)
        try:
            if name.endswith(".pdf") and now - os.path.getmtime(path) > DOWNLOAD_TTL:
                os.remove(path)
                removed += 1
        except OSError:
            logger.warning("could not remove expired report %s", name, exc_info=True)
    return removed


def verify_download(name: str, exp: int, sig: str) -> Optional[str]:
    """The file a download link names, if the link is genuine and current."""
    if exp < time.time() or not hmac.compare_digest(sig, _sign(name, exp)):
        return None
    if os.path.basename(name) != name or not name.endswith(".pdf"):
        return None
    path = os.path.join(report_dir(), name)
    return path if os.path.isfile(path) else None


def _base_url(ctx: Context) -> str:
    env = os.environ.get("APP_PUBLIC_URL", "").rstrip("/")
    if env:
        return env
    h = ctx.headers or {}
    proto = h.get("x-forwarded-proto", "https")
    host = h.get("x-forwarded-host") or h.get("host", "localhost")
    return f"{proto}://{host}"


@mcp.tool(annotations=_write("Generate a PDF report"))
@_guard
def generate_report(ctx: Context, dataset_id: str, kind: str = "analysis",
                    language: str = "en", title: str = "Data Analysis Report",
                    client_name: str = "Client", prepared_by: str = "",
                    output_path: str = "") -> str:
    """Write a client-ready PDF: `kind` "analysis" (executive summary,
    scorecard, exhibits, recommendations) or "health" (data-quality report).
    `language` is "en" or "hi". Locally the PDF is saved to `output_path`
    (default: your Downloads folder; an existing file is never overwritten,
    the report is saved beside it under a numbered name); remotely you get a
    download link valid for one hour."""
    from app import i18n
    from app.api.reports import (HealthPdfRequest, PdfRequest, _check_language,
                                 _generate_health_pdf, _generate_pdf)
    owner, local = _owner(ctx)
    df = _df(owner, dataset_id)
    meta = _store().get_meta(owner, dataset_id)
    lang = _check_language(language)
    filename = getattr(meta, "filename", "") or f"{dataset_id}.csv"
    from app.services.limits import heavy_job
    if kind == "health":
        req = HealthPdfRequest(agency_name=prepared_by or "Analytiq", language=lang)
        with heavy_job("report"), i18n.use_language(lang, **i18n.data_context(df, filename, req.agency_name)):
            pdf = _generate_health_pdf(df, filename, req)
    elif kind == "analysis":
        req = PdfRequest(title=title, client_name=client_name, prepared_by=prepared_by,
                         language=lang)
        with heavy_job("report"), i18n.use_language(lang, **i18n.data_context(df, filename, title, "",
                                                         client_name, prepared_by)):
            pdf = _generate_pdf(df, owner, dataset_id, req)
    else:
        raise ValueError('kind must be "analysis" or "health".')
    pages = len(__import__("pypdf").PdfReader(io.BytesIO(pdf)).pages)
    stem = "analytiq_{}_{}".format(kind, time.strftime("%Y%m%d_%H%M%S"))
    if local:
        target = os.path.abspath(os.path.expanduser(
            output_path or os.path.join("~", "Downloads", stem + ".pdf")))
        # Never overwrite: an existing file at the path keeps its content
        # and the report is saved beside it under a numbered name.
        base, ext = os.path.splitext(target)
        i = 1
        while os.path.exists(target):
            target = f"{base} ({i}){ext or '.pdf'}"
            i += 1
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(pdf)
        return f"Saved the {pages}-page {kind} report to `{target}`."
    name = f"{stem}_{secrets.token_hex(6)}.pdf"
    sweep_reports()
    with open(os.path.join(report_dir(), name), "wb") as f:
        f.write(pdf)
    exp = int(time.time()) + DOWNLOAD_TTL
    link = "{}/api/mcp-reports/{}?exp={}&sig={}".format(_base_url(ctx), name, exp, _sign(name, exp))
    return f"The {pages}-page {kind} report is ready: {link} (valid for one hour)."


# ── HTTP mount for the web app ───────────────────────────────────────────

def http_app():
    """The Streamable HTTP endpoint, for mounting at /mcp. Stateless: each
    request carries its own token, so one caller's session never serves
    another's."""
    return mcp.streamable_http_app(streamable_http_path="/", stateless_http=True,
                                   json_response=True)


class MCPMount:
    """The /mcp mount. The transport's session manager runs inside the web
    app's lifespan and can start only once, so each startup builds a fresh
    endpoint and this forwards to the current one."""

    def __init__(self):
        self.inner = None

    @asynccontextmanager
    async def running(self):
        self.inner = LoopbackMarker(http_app())
        async with mcp.session_manager.run():
            yield
        self.inner = None

    async def __call__(self, scope, receive, send):
        if self.inner is None:
            from starlette.responses import PlainTextResponse
            await PlainTextResponse("MCP endpoint is not running", status_code=503)(
                scope, receive, send)
            return
        await self.inner(scope, receive, send)


mount = MCPMount()


class LoopbackMarker:
    """ASGI wrapper: marks requests from this machine, so single-user open
    mode (no accounts, no admin key) serves only local callers over HTTP,
    never the internet. The header is set here and any client copy is
    removed, so it cannot be forged from outside."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = [(k, v) for k, v in scope.get("headers", [])
                       if k.lower() != b"x-analytiq-loopback"]
            client = (scope.get("client") or ("", 0))[0]
            try:
                loop = ipaddress.ip_address(client).is_loopback
            except ValueError:
                loop = client in ("testclient", "localhost")
            # Behind a reverse proxy every request arrives from the proxy,
            # often 127.0.0.1; a forwarded request is never treated as local.
            if any(k.lower() in (b"x-forwarded-for", b"forwarded", b"x-real-ip")
                   for k, _ in headers):
                loop = False
            if loop:
                headers.append((b"x-analytiq-loopback", b"1"))
            scope = dict(scope, headers=headers)
        await self.app(scope, receive, send)


def main():
    """Local entry point: `python -m app.mcp_server`, launched by Claude
    Desktop or Claude Code. Logs go to stderr; stdout carries the protocol.

    Claude starts the server from a working directory of its choosing, so
    the default `./data` would scatter datasets; local runs keep them in
    ~/.analytiq unless DATA_DIR says otherwise."""
    import sys
    if "app.config" in sys.modules:           # pragma: no cover - import order guard
        logging.getLogger(__name__).warning("config loaded before DATA_DIR default was set")
    os.environ.setdefault("DATA_DIR", os.path.join(os.path.expanduser("~"), ".analytiq"))
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
