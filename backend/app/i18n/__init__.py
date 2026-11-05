"""i18n — the language a PDF report is written in.

Report text comes from two places: the builders' own headings and labels,
and sentences the analysis engines wrote earlier (some of them cached,
like a trained model's notes, and shared with the English web app). So
text is translated where it is printed, not where it is made: every
paragraph, canvas string and chart label passes through `tr`.

A catalog maps an English template to its translation:

    "{n} rows have no {col}." -> "{n} पंक्तियों में {col} नहीं है।"

`tr` finds the template a sentence was made from, reads the values back
out of it and puts them into the translation. A value is copied as it is,
unless its placeholder is written `{name:t}`, in which case the value is
itself translated (a measure label inside a sentence, say).

The user's own data — column names, category values, the file name and
anything they typed — is never translated: `protect` registers it for the
report being built, and an exact match is left alone.

English is the identity: with the language "en" nothing is looked up and
the report is byte-for-byte what it was before this module existed.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "hi": "हिन्दी"}
DEFAULT = "en"

_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("report_lang", default=DEFAULT)
_PROTECTED: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "report_protected", default=frozenset())
# The data's column names in the report's language, by every spelling the
# reports print them in ("tenure_years", "Tenure years", "tenure years").
_COLUMNS: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "report_columns", default={})
# What `tr` produced for the report being built: text printed through two
# layers (a builder's own `tr`, then the paragraph hook) is translated once.
_DONE: contextvars.ContextVar[Optional[set]] = contextvars.ContextVar("report_done",
                                                                        default=None)
# Strings that reached `tr` with no translation, for the coverage test and
# for finding what still needs translating. Bounded so a long-running
# server cannot grow it without limit.
_MISSES: Dict[str, int] = {}
_MISS_CAP = 5000

_DIR = os.path.dirname(os.path.abspath(__file__))
_PLACEHOLDER = re.compile(r"\{(\w+)(?::(t))?\}")


def current() -> str:
    return _LANG.get()


@contextlib.contextmanager
def use_language(lang: Optional[str], protected: Iterable[str] = (),
                 columns: Iterable[str] = (), values: Iterable[str] = ()):
    """Build a report in `lang` with `protected` strings left as they are.

    Column names in `columns` whose every word the language's column
    dictionary knows are printed in the language ("salary" -> "वेतन");
    the rest, and every value, are printed as they are. A name that is
    also one of the data's `values` is left alone, so a value is never
    translated by accident."""
    lang = lang if lang in LANGUAGES else DEFAULT
    t1 = _LANG.set(lang)
    t2 = _PROTECTED.set(frozenset(_norm(p) for p in protected if isinstance(p, str) and p))
    t3 = _DONE.set(set())
    t4 = _COLUMNS.set(_column_map(lang, columns, values))
    try:
        yield
    finally:
        _LANG.reset(t1)
        _PROTECTED.reset(t2)
        _DONE.reset(t3)
        _COLUMNS.reset(t4)


def data_context(df, *extra: str) -> dict:
    """What `use_language` needs to know about a dataset."""
    values = set()
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("string", "str", "category"):
            values.update(_norm(v) for v in s.dropna().astype(str).unique()[:200_000])
    return {"protected": protected_from(df, *extra),
            "columns": [str(c) for c in df.columns], "values": values}


_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _column_map(lang: str, columns: Iterable[str], values: Iterable[str]) -> Dict[str, str]:
    if lang == DEFAULT:
        return {}
    cat = catalog(lang)
    if not cat.column_words and not cat.column_names:
        return {}
    values = {_norm(v) for v in values}
    out: Dict[str, str] = {}
    for name in columns:
        name = str(name)
        words = [w.lower() for w in _WORD.findall(name)]
        if not words:
            continue
        whole = " ".join(words)
        if whole in cat.column_names:
            target = cat.column_names[whole]
        elif all(w.isdigit() or w in cat.column_words for w in words):
            target = " ".join(t for t in (w if w.isdigit() else cat.column_words[w]
                                          for w in words) if t)
            if not target:
                continue
        else:
            continue
        label = name.replace("_", " ").strip()
        for variant in {name, label, label.lower(), label[:1].upper() + label[1:],
                        whole, whole[:1].upper() + whole[1:]}:
            v = _norm(variant)
            if v and v not in values:
                out[v] = target
    return out


def column_name(name: str) -> str:
    """A column name as the report prints it: translated when the column
    dictionary knows every word of it, otherwise unchanged."""
    return _COLUMNS.get().get(_norm(str(name)), str(name))


def protected_from(df, *extra: str, max_values: int = 200_000) -> List[str]:
    """The data's own words: column names as the reports write them, and
    the values of text columns."""
    out = []
    for c in df.columns:
        name = str(c)
        label = name.replace("_", " ").strip()
        out += [name, label, label.lower(), label[:1].upper() + label[1:]]
    for c in df.columns:
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("string", "str", "category"):
            vals = s.dropna().astype(str).unique()
            out.extend(vals[:max_values])
    out.extend(e for e in extra if e)
    return out


def _norm(s: str) -> str:
    return " ".join(str(s).split())


class _Template:
    __slots__ = ("english", "target", "regex", "names", "translate", "literal", "key")

    def __init__(self, english: str, target: str):
        self.english = english
        self.target = target
        parts, names, translate, pos = [], [], set(), 0
        for m in _PLACEHOLDER.finditer(english):
            parts.append(re.escape(english[pos:m.start()]))
            name = m.group(1)
            if name in names:
                raise ValueError(f"placeholder {{{name}}} used twice in {english!r}")
            names.append(name)
            if m.group(2):
                translate.add(name)
            # A value never runs on into the next sentence: a template
            # ending in a placeholder would otherwise swallow the rest
            # of a paragraph and pass it through untranslated.
            parts.append(f"(?P<{name}>(?:(?![.!?।]\\s+\\S).)+?)")
            pos = m.end()
        parts.append(re.escape(english[pos:]))
        self.regex = re.compile("".join(parts), re.S)
        self.names = names
        self.translate = translate
        self.literal = len(_PLACEHOLDER.sub("", english))
        first = _PLACEHOLDER.split(english, maxsplit=1)[0]
        self.key = first[:3] if len(first) >= 3 else ""
        got = {m.group(1) for m in _PLACEHOLDER.finditer(target)}
        if got != set(names):
            raise ValueError(f"placeholders differ between {english!r} and {target!r}")

    def render(self, values: Dict[str, str]) -> str:
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], self.target)


class _Catalog:
    def __init__(self, entries: Dict[str, str]):
        self.exact: Dict[str, str] = {}
        self.by_key: Dict[str, List[_Template]] = {}
        self.wild: List[_Template] = []
        self.by_english: Dict[str, _Template] = {}
        self.months: List[Tuple[re.Pattern, str]] = []
        cols = entries.get("_columns") or {}
        self.column_words: Dict[str, str] = {k.lower(): v for k, v in
                                             (cols.get("words") or {}).items()}
        self.column_names: Dict[str, str] = {" ".join(k.lower().split()): v for k, v in
                                             (cols.get("names") or {}).items()}
        for en, target in entries.items():
            if en.startswith("_"):            # comments and month tables
                continue
            en_n = _norm(en)
            if _PLACEHOLDER.search(en):
                tpl = _Template(en_n, target)
                self.by_english[en_n] = tpl
                (self.by_key.setdefault(tpl.key, []) if tpl.key else self.wild).append(tpl)
            else:
                self.exact[en_n] = target
        for name, target in (entries.get("_months") or {}).items():
            self.months.append((re.compile(r"\b%s\b" % re.escape(name)), target))

    def match(self, text: str) -> Optional[Tuple[_Template, Dict[str, str]]]:
        best = None
        for tpl in self.by_key.get(text[:3], []) + self.wild:
            m = tpl.regex.fullmatch(text)
            if m and (best is None or tpl.literal > best[0].literal):
                best = (tpl, m.groupdict())
        return best


_CATALOGS: Dict[str, _Catalog] = {}


def catalog(lang: str) -> Optional[_Catalog]:
    if lang == DEFAULT:
        return None
    if lang not in _CATALOGS:
        path = os.path.join(_DIR, f"{lang}.json")
        with open(path, encoding="utf-8") as fh:
            _CATALOGS[lang] = _Catalog(json.load(fh))
    return _CATALOGS[lang]


# A list marker or section number in front of a sentence, kept as it is.
_LEAD = re.compile(r"^(\s*(?:[•·\-–—]\s+|\d{1,2}(?:\.\d{1,2})*[.)]\s+|\d{2}\s+(?=[A-Z]))?)(.*?)(\s*)$",
                   re.S)
_STEP = re.compile(r"(?<=\S)\s+(?=\d{1,2}\.\s+[A-Z'\"])")
_DONE_SPLIT = re.compile(r"(?<=[।.!?;])\s+")
_ENTITY = re.compile(r"&\w+;|&#\d+;")
_LIST = re.compile(r"\s*(?:,|·|;)\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9'\"(])")


def tr(text, **values) -> str:
    """`text` in the report's language.

    With `values`, `text` is a template filled with them (the builders' own
    sentences); without, `text` is a finished string, matched back to the
    template it came from."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    lang = _LANG.get()
    if lang == DEFAULT:
        return _fill(text, values) if values else text
    cat = catalog(lang)
    done = _DONE.get()
    if not values and done is not None and _norm(text) in done:
        return text
    out = _tr(text, values, cat)
    if out != text:
        _check_value(out, text)
    if done is not None and len(done) < 20000:
        done.add(_norm(out))
    return out


def mark_done(*texts: str) -> None:
    """Record pieces cut from a finished translation (a caption's first
    sentence used as a title) as translations, so they are not translated
    again as if they were English."""
    done = _DONE.get()
    if done is None:
        return
    for t in texts:
        if t:
            done.add(_norm(t))


def is_done(text: str) -> bool:
    """Whether `text` is already a translation made for this report."""
    done = _DONE.get()
    if not done:
        return False
    n = _norm(_LEAD.match(text).group(2))
    if n in done or _norm(text) in done:
        return True
    # Translations joined into one caption or paragraph: the text must
    # divide, in order, into pieces each of which is a translation.
    acc = ""
    for piece in _DONE_SPLIT.split(n):
        acc = (acc + " " + piece).strip()
        if acc in done:
            acc = ""
    return acc == ""


def _tr(text: str, values: dict, cat: "_Catalog") -> str:
    if values:
        tpl = cat.by_english.get(_norm(text))
        if tpl is None:
            _miss(text)
            return _fill(text, values)
        return tpl.render({k: (_translate(str(v), cat) if k in tpl.translate
                               else _months(str(v), cat)) for k, v in values.items()})
    return _translate(text, cat)


def _fill(template: str, values: dict) -> str:
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def _translate(text: str, cat: _Catalog) -> str:
    if not text.strip():
        return text
    lead, core, tail = _LEAD.match(text).groups()
    if not core:
        return text
    out = _core(core, cat)
    if out is None and _STEP.search(core):
        # "1. Check …  2. Confirm …": each step on its own.
        steps = _STEP.split(core)
        return lead + " ".join(_translate(st, cat) for st in steps) + tail
    if out is None:
        # Engines join sentences; translate them one at a time.
        parts = _SENTENCE.split(core)
        if len(parts) > 1:
            done = [_core(p, cat) for p in parts]
            if all(d is not None for d in done):
                return lead + " ".join(done) + tail
            for p, d in zip(parts, done):
                if d is None:
                    _miss(p)
            return lead + " ".join(d if d is not None else p
                                   for p, d in zip(parts, done)) + tail
        _miss(core)
        return text
    return lead + out + tail


def _core(text: str, cat: _Catalog) -> Optional[str]:
    n = _norm(text)
    keep = _PROTECTED.get()
    cols = _COLUMNS.get()
    if n in cols:
        return cols[n]
    if n in keep or not re.search(r"[A-Za-z]", _ENTITY.sub("", n)):
        return text
    parts = [p for p in _LIST.split(n) if p]
    if len(parts) > 1 and all(p in keep for p in parts):
        return _data_value(n)             # a list of the data's own names
    if n in cat.exact:
        return cat.exact[n]
    if _ITEMS.search(n):
        listed = _items(n, cat)
        if listed is not None:
            return listed
    hit = cat.match(n)
    if hit is None:
        return None
    tpl, vals = hit
    out = {}
    for k, v in vals.items():
        if k in tpl.translate:
            t = _core(v, cat)
            if t is None:
                _miss(v)
                t = v
            out[k] = t
        else:
            out[k] = _months(v, cat)
    return tpl.render(out)


# Latin-script words a translation may keep: names and notation.
_KEPT_WORDS = {"SQL", "Analytiq", "Unknown", "Gini", "Kruskal", "Wallis", "Mann",
               "Whitney", "Spearman", "Pearson", "Iglewicz", "Hoaglin", "COGS",
               "RFM", "IQR", "ROC", "AUC", "MAE", "RMSE", "XGBoost", "True", "False"}


def _check_value(out: str, source: str) -> None:
    """Record a finished translation that still carries English words. A
    placeholder matches any text, so "{v4}" could take "16.4M over Dec 2009
    to Dec 2011" whole and the sentence came out half English with nothing
    recorded. The data's own names and values are allowed."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", _ENTITY.sub("", out))
             if w not in _KEPT_WORDS]
    if not words or _only_data(" ".join(words)):
        return
    _miss(source)


_QUOTED = re.compile(r"'([^']+)'")
_ITEMS = re.compile(r"(\s+·\s+)")


def _items(text: str, cat: _Catalog) -> Optional[str]:
    """"0–2 years 30% (n=73) · 3–5 years 22% (n=82)": a list whose items
    are translated one by one; items that are the data's own words stay."""
    pieces = _ITEMS.split(text)
    if len(pieces) < 3:
        return None
    out, changed = [], False
    for i, piece in enumerate(pieces):
        if i % 2 or not piece:
            out.append(piece)
            continue
        t = _core(piece, cat)
        if t is None:
            if not _only_data(piece):
                return None
            t = _data_value(piece)
        changed |= t != piece
        out.append(t)
    return "".join(out) if changed else None


_DATA_WORDS: Dict[frozenset, frozenset] = {}


def _only_data(text: str) -> bool:
    """Whether every word in `text` is one of the data's own."""
    keep = _PROTECTED.get()
    words = _DATA_WORDS.get(keep)
    if words is None:
        _DATA_WORDS.clear()
        words = _DATA_WORDS[keep] = frozenset(
            w for p in keep for w in re.findall(r"[A-Za-z]+", p))
    return all(w in words for w in re.findall(r"[A-Za-z]+", text))


def _data_value(value: str) -> str:
    """A copied value with the data's column names in the report's
    language: on its own, in quotes, or in a list."""
    cols = _COLUMNS.get()
    if not cols:
        return value
    n = _norm(value)
    if n in cols:
        return cols[n]
    value = _QUOTED.sub(lambda m: "'" + cols.get(_norm(m.group(1)), m.group(1)) + "'", value)
    parts = _LIST.split(n)
    if len(parts) > 1 and all(p in cols or p in _PROTECTED.get() for p in parts if p):
        seps = _LIST.findall(n)
        out = [cols.get(p, p) for p in parts]
        return "".join(a + (seps[i] if i < len(seps) else "") for i, a in enumerate(out))
    return value


def _months(value: str, cat: _Catalog) -> str:
    """A copied value: the data's column names in the report's language,
    and month and weekday names ("Mar 2024")."""
    value = _data_value(value)
    if _norm(value) in _PROTECTED.get():
        return value
    for rx, target in cat.months:
        value = rx.sub(target, value)
    return value


def date_text(d) -> str:
    """A date as the report's language writes it."""
    lang = _LANG.get()
    if lang == DEFAULT:
        return d.strftime("%B %d, %Y")
    cat = catalog(lang)
    return _months(f"{d.day} {d.strftime('%B')} {d.year}", cat)


def month_year(d) -> str:
    """"Mar 2024", in the report's language."""
    text = d.strftime("%b %Y")
    return text if _LANG.get() == DEFAULT else _months(text, catalog(_LANG.get()))


def _miss(text: str):
    n = _norm(text)
    if n in _MISSES or len(_MISSES) < _MISS_CAP:
        _MISSES[n] = _MISSES.get(n, 0) + 1


def misses() -> Dict[str, int]:
    return dict(_MISSES)


def clear_misses():
    _MISSES.clear()
