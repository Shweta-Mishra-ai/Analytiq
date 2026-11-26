"""
Email verification, password reset by email, and keeping one person from
collecting free credits on many accounts. Mail is captured, not sent.
"""
from __future__ import annotations

import re

import pytest

from app.config import config
from app.services import emails
from app.services.credits import credit_store
from app.services.user_store import user_store

PASSWORD = "plenty-long-pw-9"


@pytest.fixture()
def mail(client, tmp_path, monkeypatch):
    from app import main
    outbox = []
    monkeypatch.setattr(emails, "_deliver", lambda msg: outbox.append(msg))
    monkeypatch.setattr(config, "smtp_host", "smtp.test")
    monkeypatch.setattr(config, "require_email_verification", True)
    monkeypatch.setattr(config, "app_admin_key", "email-admin-key")
    monkeypatch.setattr(config, "public_url", "https://app.test")
    monkeypatch.setattr(config, "free_daily_credits", 5)
    monkeypatch.setattr(main, "signup_limiter", main._SignupLimiter(per_hour=50))
    credit_store.open(str(tmp_path / "billing"))
    names = []
    yield outbox, names
    for n in names:
        user_store.delete(n)
    credit_store.open()


def _link_token(msg, path):
    body = msg.get_content()
    m = re.search(r"https://app\.test/" + path + r"\?token=(\S+)", body)
    assert m, body
    return m.group(1)


def _signup(client, names, name, email):
    names.append(name)
    return client.post("/api/auth/signup", json={"username": name, "password": PASSWORD,
                                                 "email": email})


def test_free_credits_start_only_after_the_email_is_confirmed(client, mail):
    outbox, names = mail
    r = _signup(client, names, "verify_me", "verify.me@example.com")
    assert r.status_code == 200 and r.json()["verification_sent"]
    h = {"Authorization": "Bearer " + r.json()["token"]}
    bal = client.get("/api/account", headers=h).json()["credits"]
    assert bal["free_left"] == 0 and bal["needs_verification"]
    assert len(outbox) == 1 and outbox[0]["To"] == "verify.me@example.com"
    tok = _link_token(outbox[0], "verify-email")
    assert client.post("/api/auth/verify-email", json={"token": tok}).status_code == 200
    bal = client.get("/api/account", headers=h).json()["credits"]
    assert bal["free_left"] == 5 and not bal["needs_verification"]


def test_unverified_account_is_told_to_confirm_not_to_buy(client, mail, hr_csv_bytes):
    import io
    outbox, names = mail
    r = _signup(client, names, "unverified", "unverified@example.com")
    h = {"Authorization": "Bearer " + r.json()["token"]}
    ds = client.post("/api/datasets/upload", headers=h,
                     files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")}).json()
    out = client.post("/api/reports/{}/pdf".format(ds["meta"]["dataset_id"]), headers=h, json={})
    assert out.status_code == 402 and "Confirm your email" in out.json()["detail"]


def test_one_mailbox_one_account_including_gmail_aliases(client, mail):
    outbox, names = mail
    assert _signup(client, names, "first_one", "Some.Person@gmail.com").status_code == 200
    for alias in ("someperson@gmail.com", "some.person+free@googlemail.com",
                  "SOMEPERSON@GMAIL.COM"):
        r = _signup(client, names, "again_" + str(len(names)), alias)
        assert r.status_code == 422 and "already exists" in r.json()["detail"], alias


def test_throwaway_inboxes_are_refused(client, mail):
    outbox, names = mail
    r = _signup(client, names, "burner", "x@mailinator.com")
    assert r.status_code == 422 and "temporary" in r.json()["detail"]


def test_with_email_on_an_address_is_required(client, mail):
    outbox, names = mail
    r = _signup(client, names, "noemail", "")
    assert r.status_code == 422


def test_a_verification_link_is_not_a_sign_in_token(client, mail):
    outbox, names = mail
    _signup(client, names, "linkuser", "link.user@example.com")
    tok = _link_token(outbox[0], "verify-email")
    r = client.get("/api/datasets", headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 401


def test_password_reset_by_email_works_once(client, mail):
    outbox, names = mail
    _signup(client, names, "forgetful", "forgetful@example.com")
    outbox.clear()
    r = client.post("/api/auth/forgot-password", json={"login": "forgetful@example.com"})
    assert r.status_code == 200 and len(outbox) == 1
    tok = _link_token(outbox[0], "reset-password")
    r = client.post("/api/auth/reset-password", json={"token": tok, "password": "a-new-strong-pw-2"})
    assert r.status_code == 200 and r.json()["token"]
    assert user_store.verify("forgetful", "a-new-strong-pw-2")
    again = client.post("/api/auth/reset-password", json={"token": tok, "password": "another-pw-33"})
    assert again.status_code == 400, "a used reset link worked a second time"
    # Opening the mailed link proved the mailbox.
    assert user_store.get("forgetful").email_verified


def test_forgot_password_does_not_reveal_who_has_an_account(client, mail):
    outbox, names = mail
    _signup(client, names, "known_user", "known@example.com")
    outbox.clear()
    a = client.post("/api/auth/forgot-password", json={"login": "known_user"})
    b = client.post("/api/auth/forgot-password", json={"login": "nobody_here"})
    assert a.status_code == b.status_code == 200 and a.json() == b.json()
    assert len(outbox) == 1


def test_without_smtp_signup_works_as_before(client, mail, monkeypatch):
    outbox, names = mail
    monkeypatch.setattr(config, "smtp_host", "")
    r = _signup(client, names, "no_smtp", "")
    assert r.status_code == 200
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert client.get("/api/account", headers=h).json()["credits"]["free_left"] == 5
    assert client.post("/api/auth/forgot-password", json={"login": "no_smtp"}).status_code == 503


def test_admin_created_and_older_accounts_keep_their_free_credits(client, mail):
    outbox, names = mail
    names.append("by_admin")
    r = client.post("/api/admin/users", headers={"Authorization": "Bearer email-admin-key"},
                    json={"username": "by_admin", "password": PASSWORD})
    assert r.status_code == 200
    assert credit_store.balance("by_admin")["free_left"] == 5


def test_email_normalisation():
    n = emails.normalize_email
    assert n("A.B+x@GMail.com") == n("ab@gmail.com") == "ab@gmail.com"
    assert n("a.b@example.com") != n("ab@example.com")
    assert n("not-an-email") == ""


def test_a_purchase_emails_the_receipt_as_a_pdf(client, mail, tmp_path):
    outbox, names = mail
    from app.services.invoices import invoice_book, issue_and_send
    invoice_book.open(str(tmp_path / "inv"))
    try:
        _signup(client, names, "buyer_mail", "buyer@example.com")
        outbox.clear()
        inv = issue_and_send(user="buyer_mail", description="50 credits — Starter pack",
                             credits=50, amount=19900, currency="INR", provider="razorpay",
                             ref="razorpay:order_X")
        assert len(outbox) == 1 and inv["number"] in outbox[0]["Subject"]
        files = [p for p in outbox[0].iter_attachments()]
        assert files and files[0].get_content()[:4] == b"%PDF"
    finally:
        invoice_book.open()
