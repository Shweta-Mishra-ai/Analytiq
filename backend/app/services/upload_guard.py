"""
services/upload_guard.py — what an uploaded file is, judged by its bytes.

Every upload path (dataset files, image and video extraction, knowledge-base
files, the MCP server) calls `check()` before parsing anything. The name a
browser sends proves nothing: an executable renamed "sales.csv" or a
macro-bearing workbook renamed ".xlsx" used to reach the parsers.

Always on:
  * the content must match the extension (CSV/TXT/JSON are text with no
    NUL bytes or executable headers; .xlsx is a zip with a workbook in it;
    .xls an OLE file; images and videos carry their format's signature);
  * workbooks with macros (a VBA project) are refused: Analytiq never runs
    them, but it should not store and pass on a file built to carry code;
  * zip bombs are refused before pandas expands them: too many entries,
    too much uncompressed data, or an implausible compression ratio;
  * encrypted (password-protected) workbooks are refused with a clear
    message instead of a parser error.

Optional antivirus: when CLAMD_HOST (and optionally CLAMD_PORT, default
3310) or CLAMD_SOCKET is set, every upload is also scanned by a ClamAV
daemon over its INSTREAM protocol. If the daemon is configured but cannot
be reached, uploads are refused (fail closed) rather than let through
unscanned.
"""
from __future__ import annotations

import io
import logging
import os
import socket
import struct
import zipfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Zip-bomb limits. A 50 MB workbook legitimately expands to a few hundred MB
# of XML; these leave room for that and stop the 42 KB -> 4 PB kind.
MAX_UNZIPPED_BYTES = int(os.environ.get("MAX_UNZIPPED_MB", "1024")) * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
MAX_RATIO = 200                     # uncompressed / compressed, per entry

_EXECUTABLE_MAGIC = (
    b"MZ",                          # Windows PE
    b"\x7fELF",                     # Linux ELF
    b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe",  # Mach-O
    b"#!",                          # script with an interpreter line
)
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP = b"PK\x03\x04"

_IMAGE_MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".bmp": (b"BM",),
    ".webp": (b"RIFF",),
}
_TEXT_EXTS = {".csv", ".tsv", ".txt", ".md", ".log", ".json"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


class UploadRejected(Exception):
    """The upload is refused. `status` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class ScanResult:
    scanned: bool                   # True when ClamAV looked at the file
    engine: str = ""


def check(filename: str, data: bytes) -> ScanResult:
    """Refuse `data` unless it is the kind of file `filename` says it is,
    carries no macros or zip bomb, and (when ClamAV is configured) is clean.
    Raises UploadRejected."""
    ext = os.path.splitext((filename or "").lower())[1]
    if not data:
        raise UploadRejected("The file is empty.")
    if data.startswith(_EXECUTABLE_MAGIC):
        raise UploadRejected("This file is a program or script, not data, and was refused.")

    if ext in _TEXT_EXTS:
        _check_text(data, ext)
    elif ext == ".xlsx":
        _check_xlsx(data)
    elif ext == ".xls":
        _check_xls(data)
    elif ext in _IMAGE_MAGIC:
        if not data.startswith(_IMAGE_MAGIC[ext]) or (
                ext == ".webp" and data[8:12] != b"WEBP"):
            raise UploadRejected(f"The file is not a {ext[1:].upper()} image.")
    elif ext in _VIDEO_EXTS:
        _check_video(data, ext)
    # Other extensions are refused by the parser behind each route; this
    # module only vouches for content.

    return _antivirus(filename, data)


# ── content checks ────────────────────────────────────────

def _check_text(data: bytes, ext: str) -> None:
    head = data[:65536]
    if data.startswith((_ZIP, _OLE, b"%PDF")):
        raise UploadRejected(
            f"The file is named {ext} but is a binary document (a workbook, "
            "archive or PDF). Save it as {} and upload again.".format(ext))
    # UTF-16 text has NULs between letters; it starts with a byte-order mark.
    if b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UploadRejected(f"The file is named {ext} but is not text.")


def _check_xlsx(data: bytes) -> None:
    if data.startswith(_OLE):
        # Excel saves a password-protected .xlsx as an OLE container.
        raise UploadRejected(
            "This workbook is password-protected. Remove the password in Excel "
            "and upload it again.")
    if not data.startswith(_ZIP):
        raise UploadRejected("The file is named .xlsx but is not an Excel workbook.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except zipfile.BadZipFile:
        raise UploadRejected("The workbook is damaged and cannot be opened.")
    if len(infos) > MAX_ZIP_ENTRIES:
        raise UploadRejected("The workbook has an implausible number of parts and was refused.")
    total = 0
    names = set()
    for info in infos:
        names.add(info.filename.lower())
        if info.flag_bits & 0x1:
            raise UploadRejected(
                "This workbook is encrypted. Remove the password and upload it again.")
        total += info.file_size
        if info.compress_size and info.file_size / info.compress_size > MAX_RATIO \
                and info.file_size > 50 * 1024 * 1024:
            raise UploadRejected("The workbook expands to an implausible size "
                                 "(a compression bomb) and was refused.")
    if total > MAX_UNZIPPED_BYTES:
        raise UploadRejected("The workbook expands to more than {} MB and was refused."
                             .format(MAX_UNZIPPED_BYTES // (1024 * 1024)))
    if "[content_types].xml" not in names or not any(
            n.startswith("xl/") for n in names):
        raise UploadRejected("The file is named .xlsx but is not an Excel workbook.")
    if any(n.endswith("vbaproject.bin") for n in names):
        raise UploadRejected(
            "This workbook contains macros. Save it as a plain .xlsx "
            "(Excel Workbook, not macro-enabled) and upload it again.")


def _check_xls(data: bytes) -> None:
    if not data.startswith(_OLE):
        raise UploadRejected("The file is named .xls but is not an Excel workbook.")
    # The VBA storage is named "_VBA_PROJECT" (UTF-16 in the OLE directory).
    if "_VBA_PROJECT".encode("utf-16-le") in data or b"_VBA_PROJECT" in data:
        raise UploadRejected(
            "This workbook contains macros. Save it as .xlsx (Excel Workbook) "
            "and upload it again.")


def _check_video(data: bytes, ext: str) -> None:
    ok = {
        ".mp4": data[4:8] == b"ftyp",
        ".mov": data[4:8] in (b"ftyp", b"moov", b"wide", b"mdat", b"free"),
        ".webm": data.startswith(b"\x1a\x45\xdf\xa3"),
        ".mkv": data.startswith(b"\x1a\x45\xdf\xa3"),
        ".avi": data.startswith(b"RIFF") and data[8:12] == b"AVI ",
    }[ext]
    if not ok:
        raise UploadRejected(f"The file is not a {ext[1:].upper()} video.")


# ── ClamAV ────────────────────────────────────────────────

def clamav_configured() -> bool:
    return bool(os.environ.get("CLAMD_HOST") or os.environ.get("CLAMD_SOCKET"))


def _connect() -> socket.socket:
    path = os.environ.get("CLAMD_SOCKET")
    if path:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(float(os.environ.get("CLAMD_TIMEOUT", "60")))
        s.connect(path)
        return s
    host = os.environ["CLAMD_HOST"]
    port = int(os.environ.get("CLAMD_PORT", "3310"))
    return socket.create_connection((host, port),
                                    timeout=float(os.environ.get("CLAMD_TIMEOUT", "60")))


def clamd_scan(data: bytes) -> Optional[str]:
    """The signature name if ClamAV finds one, None if the file is clean.
    Raises OSError when the daemon cannot be reached or answers oddly."""
    with _connect() as s:
        s.sendall(b"zINSTREAM\0")
        view = memoryview(data)
        for i in range(0, len(data), 1 << 20):
            chunk = view[i:i + (1 << 20)]
            s.sendall(struct.pack("!L", len(chunk)) + chunk)
        s.sendall(struct.pack("!L", 0))
        reply = b""
        while not reply.endswith(b"\0"):
            part = s.recv(4096)
            if not part:
                break
            reply += part
    text = reply.rstrip(b"\0").decode("utf-8", "replace")
    if text.endswith("OK"):
        return None
    if text.endswith("FOUND"):
        return text.split(":", 1)[-1].rsplit(" ", 1)[0].strip()
    raise OSError(f"unexpected ClamAV reply: {text[:200]}")


def _antivirus(filename: str, data: bytes) -> ScanResult:
    if not clamav_configured():
        return ScanResult(scanned=False)
    try:
        found = clamd_scan(data)
    except OSError:
        logger.error("ClamAV scan failed for %r", filename, exc_info=True)
        # Fail closed: an antivirus that is configured but down must not
        # quietly turn into no antivirus.
        raise UploadRejected("The virus scanner is unavailable, so the upload was "
                             "refused. Try again shortly.", status=503)
    if found:
        logger.warning("ClamAV rejected %r: %s", filename, found)
        raise UploadRejected(f"The file was flagged by the virus scanner ({found}) "
                             "and was refused.")
    return ScanResult(scanned=True, engine="ClamAV")
