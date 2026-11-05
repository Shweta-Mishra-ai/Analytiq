"""
Uploads are judged by their bytes: renamed executables, macro workbooks,
zip bombs and encrypted workbooks are refused before any parser sees them,
and an optional ClamAV daemon scans everything when configured.
"""
from __future__ import annotations

import io
import socket
import struct
import threading
import zipfile

import pandas as pd
import pytest

from app.services import upload_guard as ug
from app.services.upload_guard import UploadRejected, check


def _xlsx(extra=None) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(buf, index=False)
    if not extra:
        return buf.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as src, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        for name, data in extra.items():
            dst.writestr(name, data)
    return out.getvalue()


def test_ordinary_files_pass():
    check("sales.csv", b"region,revenue\nNorth,10\n")
    check("sales.json", b'[{"a": 1}]')
    check("book.xlsx", _xlsx())
    check("chart.png", b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    check("clip.mp4", b"\0\0\0\x18ftypmp42" + b"\0" * 32)
    check("utf16.csv", "a,b\n1,2\n".encode("utf-16"))


@pytest.mark.parametrize("data", [b"MZ\x90\x00rest-of-a-windows-program",
                                  b"\x7fELF\x02\x01\x01", b"#!/bin/sh\nrm -rf /\n"])
def test_programs_renamed_as_data_are_refused(data):
    with pytest.raises(UploadRejected, match="program or script"):
        check("sales.csv", data)


def test_binary_named_as_csv_is_refused():
    with pytest.raises(UploadRejected, match="binary document"):
        check("sales.csv", _xlsx())
    with pytest.raises(UploadRejected, match="not text"):
        check("sales.csv", b"a,b\n\x00\x01\x02\x03")


def test_macro_workbooks_are_refused():
    with pytest.raises(UploadRejected, match="macros"):
        check("book.xlsx", _xlsx({"xl/vbaProject.bin": b"\0" * 64}))
    ole = ug._OLE + b"\0" * 64 + "_VBA_PROJECT".encode("utf-16-le")
    with pytest.raises(UploadRejected, match="macros"):
        check("old.xls", ole)


def test_password_protected_workbook_gets_a_clear_message():
    with pytest.raises(UploadRejected, match="password"):
        check("book.xlsx", ug._OLE + b"\0" * 512)


def test_a_compression_bomb_is_refused_before_parsing(monkeypatch):
    monkeypatch.setattr(ug, "MAX_UNZIPPED_BYTES", 5 * 1024 * 1024)
    bomb = _xlsx({"xl/worksheets/huge.xml": b"0" * (6 * 1024 * 1024)})
    assert len(bomb) < 200_000           # tiny on disk
    with pytest.raises(UploadRejected, match="expands"):
        check("book.xlsx", bomb)


def test_a_zip_that_is_not_a_workbook_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("payload.exe", b"MZ")
    with pytest.raises(UploadRejected, match="not an Excel workbook"):
        check("book.xlsx", buf.getvalue())


def test_an_image_that_is_not_an_image_is_refused():
    with pytest.raises(UploadRejected, match="PNG"):
        check("chart.png", b"<svg onload=alert(1)>")


def test_the_upload_route_refuses_with_the_reason(client):
    r = client.post("/api/datasets/upload",
                    files={"file": ("sales.csv", b"MZ\x90\x00", "text/csv")})
    assert r.status_code == 422
    assert "program or script" in r.json()["detail"]


# ── ClamAV ────────────────────────────────────────────────

class _FakeClamd:
    """Speaks the INSTREAM protocol; flags anything containing EICAR."""

    def __init__(self, reply_ok=True):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.received = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                cmd = b""
                while not cmd.endswith(b"\0"):
                    cmd += conn.recv(1)
                data = b""
                while True:
                    size = struct.unpack("!L", self._read(conn, 4))[0]
                    if size == 0:
                        break
                    data += self._read(conn, size)
                self.received.append(data)
                verdict = (b"stream: Eicar-Test-Signature FOUND\0" if b"EICAR" in data
                           else b"stream: OK\0")
                conn.sendall(verdict)

    @staticmethod
    def _read(conn, n):
        buf = b""
        while len(buf) < n:
            buf += conn.recv(n - len(buf))
        return buf

    def close(self):
        self.sock.close()


def test_clamav_scans_and_rejects_when_configured(monkeypatch):
    fake = _FakeClamd()
    monkeypatch.setenv("CLAMD_HOST", "127.0.0.1")
    monkeypatch.setenv("CLAMD_PORT", str(fake.port))
    try:
        assert check("sales.csv", b"a,b\n1,2\n").scanned
        assert fake.received[-1] == b"a,b\n1,2\n"
        with pytest.raises(UploadRejected, match="Eicar-Test-Signature"):
            check("sales.csv", b"a,b\nEICAR,1\n")
    finally:
        fake.close()


def test_a_configured_scanner_that_is_down_fails_closed(monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                   # nothing listens there now
    monkeypatch.setenv("CLAMD_HOST", "127.0.0.1")
    monkeypatch.setenv("CLAMD_PORT", str(port))
    with pytest.raises(UploadRejected) as e:
        check("sales.csv", b"a,b\n1,2\n")
    assert e.value.status == 503


def test_without_clamav_nothing_is_scanned(monkeypatch):
    monkeypatch.delenv("CLAMD_HOST", raising=False)
    monkeypatch.delenv("CLAMD_SOCKET", raising=False)
    assert not check("sales.csv", b"a,b\n1,2\n").scanned
