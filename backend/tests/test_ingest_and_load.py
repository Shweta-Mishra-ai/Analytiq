"""
The two ways an app fails a user without a bug in the analysis: it
refuses a file it could have read, and it falls over when several people
use it at once.

**Errors that name the library, not the problem.** A fuzz pass over the
ingest paths — RAG extraction, image and video table extraction, token
verification — raised `PdfiumError: Data format error`, `BadZipFile: File
is not a zip file`, `EmptyDataError: No columns to parse from file` and
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff`. Every one of
those names the parser's difficulty and none says what the person who
uploaded it should do.

**Information lost at the door.** Reading every CSV as UTF-8 rejected
files Excel on Windows produces, which are cp1252; decoding with
`errors="replace"` turned every accented character into U+FFFD. A file
that was perfectly readable came back either refused or corrupted.

**No limit on concurrent work.** One ML pipeline is about ten seconds
and 250MB on this machine, dispatched to an unbounded threadpool. Ten
people pressing "train" together is 2.5GB and an out-of-memory kill —
and an OOM kill takes down every request in flight, not only the ones
that caused it.
"""
from __future__ import annotations

import threading
import time

import pytest


# ══════════════════════════════════════════════════════════
#  Every refusal is addressed to the user
# ══════════════════════════════════════════════════════════

BAD_BLOBS = {
    "empty": b"",
    "one_byte": b"\x00",
    "binary": b"\xff\xfe\x00\x01binary garbage\x80\x81",
    "html_not_csv": b"<html><body>Not a spreadsheet</body></html>",
    "truncated_pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog",
    "truncated_png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,
    "control_chars": bytes(range(0, 32)) * 200,
}

# The parser exceptions that must never reach a user. Each names the
# library's problem rather than theirs.
LEAKY = ("PdfiumError", "BadZipFile", "EmptyDataError", "UnicodeDecodeError",
         "ParserError", "KeyError", "AttributeError", "IndexError",
         "TypeError", "ZeroDivisionError", "OSError")


@pytest.mark.parametrize("blob", sorted(BAD_BLOBS))
@pytest.mark.parametrize("name", ["x.pdf", "x.docx", "x.csv", "x.tsv", "x.txt"])
def test_no_parser_exception_reaches_the_user(name, blob):
    from app.rag import extractors

    try:
        extractors.extract(name, BAD_BLOBS[blob])
    except Exception as exc:
        assert type(exc).__name__ not in LEAKY, (
            name, blob, type(exc).__name__, str(exc)[:120])


@pytest.mark.parametrize("name", ["x.pdf", "x.docx"])
def test_a_refusal_says_what_to_do_about_it(name):
    from app.rag import extractors

    with pytest.raises(Exception) as caught:
        extractors.extract(name, b"\xff\xfe garbage")
    message = str(caught.value)
    assert name in message, message
    # Something the reader can act on, not just a statement of failure.
    assert any(w in message.lower() for w in
               ("check", "re-save", "remove", "upload again", "empty")), message


def test_a_csv_of_odd_bytes_is_read_rather_than_refused():
    """Refusing is the last resort, not the first. With the encoding
    fallback in place a byte sequence that is not UTF-8 still decodes,
    and pandas skips the lines it cannot parse — so the user gets
    whatever was readable instead of an error about the whole file. If
    nothing at all comes back, that is a refusal with a message, not a
    traceback."""
    from app.rag import extractors

    try:
        passages = extractors.extract("x.csv", b"\xff\xfe garbage")
    except Exception as exc:
        assert type(exc).__name__ not in LEAKY, type(exc).__name__
        assert "x.csv" in str(exc)
        return
    assert isinstance(passages, list)


def test_an_empty_upload_says_so_plainly():
    from app.rag import extractors

    with pytest.raises(Exception) as caught:
        extractors.extract("report.pdf", b"")
    assert "empty" in str(caught.value).lower()


def test_an_unsupported_type_is_still_a_clear_refusal():
    from app.rag import extractors

    with pytest.raises(ValueError, match="Unsupported file type"):
        extractors.extract("x.exe", b"MZ\x90\x00")


# ══════════════════════════════════════════════════════════
#  Nothing readable is thrown away
# ══════════════════════════════════════════════════════════

def test_a_windows_csv_is_read_not_rejected():
    """Excel on Windows writes cp1252. Reading it as UTF-8 raised
    outright, and `errors="replace"` turned every accented character
    into U+FFFD — information lost on a file that was perfectly
    readable."""
    from app.rag.extractors import decode_text

    data = "café,naïve\nMünchen,Ångström\n".encode("cp1252")
    text = decode_text(data)
    assert "�" not in text, text
    assert "café" in text and "München" in text


def test_a_utf8_bom_does_not_leak_into_the_first_column():
    from app.rag.extractors import decode_text

    text = decode_text("﻿name,value\na,1\n".encode("utf-8"))
    assert text.lstrip("﻿").startswith("name")


def test_plain_utf8_still_round_trips():
    from app.rag.extractors import decode_text

    original = "héllo wörld — 中文 עברית ✓\n"
    assert decode_text(original.encode("utf-8")) == original


def test_undecodable_bytes_degrade_rather_than_raise():
    from app.rag.extractors import decode_text

    assert isinstance(decode_text(bytes(range(256))), str)


# ══════════════════════════════════════════════════════════
#  A malformed passage does not sink the document
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("passage", [
    {}, {"text": None}, {"text": 42}, {"text": ""}, {"text": "   "},
    "not a dict", None,
])
def test_a_bad_passage_is_skipped_not_raised(passage):
    """A blank PDF page or an image with nothing recognised used to raise
    KeyError here and take the whole ingest down. One unreadable page is
    not a reason to reject the document."""
    from app.rag.service import chunk_passages

    assert chunk_passages([passage]) == []


def test_good_passages_survive_a_bad_neighbour():
    from app.rag.service import chunk_passages

    chunks = chunk_passages([
        {"text": None},
        {"text": "A real passage with content.", "source": "f", "locator": "1"},
        {},
    ])
    assert len(chunks) == 1
    assert chunks[0]["source"] == "f"


# ══════════════════════════════════════════════════════════
#  Load
# ══════════════════════════════════════════════════════════

def test_the_guard_admits_up_to_its_limit():
    from app.services.load_guard import _Guard

    guard = _Guard(2, "test")
    with guard.slot():
        with guard.slot():
            assert guard.running == 2


def test_the_guard_refuses_beyond_its_limit():
    from app.services.load_guard import Busy, _Guard

    guard = _Guard(1, "test")
    with guard.slot():
        with pytest.raises(Busy):
            with guard.slot():
                pass


def test_a_refusal_tells_the_user_nothing_was_lost():
    """Someone turned away needs to know it is worth retrying, and that
    their data is still there."""
    from app.services.load_guard import Busy, _Guard

    guard = _Guard(1, "model training")
    with guard.slot():
        with pytest.raises(Busy) as caught:
            with guard.slot():
                pass
    message = str(caught.value)
    assert "nothing has been lost" in message.lower(), message
    assert "try again" in message.lower(), message


def test_a_slot_is_released_when_the_work_raises():
    """A job that fails must not hold its slot forever, or the limit
    ratchets down to zero over a day."""
    from app.services.load_guard import _Guard

    guard = _Guard(1, "test")
    with pytest.raises(ValueError):
        with guard.slot():
            raise ValueError("boom")
    assert guard.running == 0
    with guard.slot():
        pass


def test_the_guard_holds_under_real_threads():
    from app.services.load_guard import Busy, _Guard

    guard = _Guard(3, "test")
    admitted, refused = [], []
    peak = []

    def worker():
        try:
            with guard.slot():
                peak.append(guard.running)
                admitted.append(1)
                time.sleep(0.05)
        except Busy:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(admitted) + len(refused) == 20
    assert max(peak) <= 3, max(peak)
    assert guard.running == 0


def test_the_limits_are_configurable():
    """A bigger container should be able to raise them without a code
    change."""
    from app.config import config

    assert config.max_concurrent_training >= 1
    assert config.max_concurrent_analysis >= 1


def test_the_snapshot_reports_what_is_running():
    from app.services.load_guard import TRAINING, snapshot

    with TRAINING.slot():
        state = snapshot()
    assert state["training"]["running"] == 1
    assert state["training"]["limit"] == TRAINING.limit
    assert snapshot()["training"]["running"] == 0
