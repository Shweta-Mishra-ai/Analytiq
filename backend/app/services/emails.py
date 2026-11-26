"""
services/emails.py — sending email, and telling email addresses apart.

Mail goes out through any SMTP server (Gmail, Zoho, Brevo, Amazon SES,
your host's): SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.
Without SMTP_HOST nothing is sent and the features that need mail say so:
sign-up then works without verification, as before, and a lost password is
reset by the administrator.

Free daily credits made every new account worth something, so one person
could open many. Each account now needs its own verified mailbox, and two
ways of writing the same mailbox count as one: case, and Gmail's dots and
"+tags" (a.b+1@gmail.com is ab@gmail.com). Throwaway-mail domains are
refused.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

from fastapi import HTTPException

from app.config import config

logger = logging.getLogger(__name__)

_GMAIL = {"gmail.com", "googlemail.com"}

# The commonest throwaway-inbox services. Not exhaustive: the point is to
# make a stream of free accounts take real effort.
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "sharklasers.com",
    "10minutemail.com", "10minutemail.net", "tempmail.com", "temp-mail.org",
    "temp-mail.io", "tempmail.net", "yopmail.com", "yopmail.net", "trashmail.com",
    "getnada.com", "nada.email", "dispostable.com", "maildrop.cc", "mailnesia.com",
    "fakeinbox.com", "throwawaymail.com", "mohmal.com", "emailondeck.com",
    "mintemail.com", "mytemp.email", "tempail.com", "burnermail.io", "moakt.com",
    "spamgourmet.com", "mailcatch.com", "tempr.email", "discard.email",
    "inboxkitten.com", "1secmail.com", "1secmail.net", "1secmail.org",
    "emailfake.com", "tmpmail.org", "tmpmail.net", "minuteinbox.com", "trash-mail.com",
}


def normalize_email(email: str) -> str:
    """The mailbox an address delivers to, for telling accounts apart."""
    email = (email or "").strip().lower()
    if email.count("@") != 1:
        return ""
    local, domain = email.split("@")
    if domain in _GMAIL:
        local = local.split("+", 1)[0].replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}" if local and domain else ""


def check_address(email: str) -> str:
    """The address, stripped, or HTTPException 422 saying what is wrong."""
    email = (email or "").strip()
    if not email:
        raise HTTPException(422, "An email address is required.")
    if (len(email) > 200 or email.count("@") != 1 or " " in email
            or "." not in email.split("@")[1]):
        raise HTTPException(422, "That email address does not look right.")
    if email.split("@")[1].lower() in DISPOSABLE_DOMAINS:
        raise HTTPException(422, "Please use a permanent email address, not a "
                                 "temporary inbox.")
    return email


def enabled() -> bool:
    return bool(config.smtp_host)


def verification_required() -> bool:
    """New sign-ups must confirm their address before free credits."""
    return enabled() and config.require_email_verification


def _deliver(msg: EmailMessage) -> None:
    """Hand a message to the SMTP server (the tests replace this)."""
    context = ssl.create_default_context()
    if config.smtp_port == 465:
        server = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=15,
                                  context=context)
    else:
        server = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15)
    with server:
        if config.smtp_port != 465 and config.smtp_starttls:
            server.starttls(context=context)
        if config.smtp_user:
            server.login(config.smtp_user, config.smtp_password)
        server.send_message(msg)


def send(to: str, subject: str, text: str,
         attachment: Optional[tuple[str, bytes, str]] = None) -> None:
    """Send one plain-text email. Raises HTTPException 502 if the server
    will not take it, so the person waiting is told."""
    if not enabled():
        raise HTTPException(503, "Email is not set up on this server.")
    msg = EmailMessage()
    msg["From"] = config.smtp_from or config.smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    if attachment:
        name, data, mime = attachment
        maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    try:
        _deliver(msg)
    except (smtplib.SMTPException, OSError) as e:
        logger.error("email to %s failed: %s", to.split("@")[-1], e)
        raise HTTPException(502, "The email could not be sent. Please try again in "
                                 "a few minutes.")


def _sign_off() -> str:
    who = config.business_name or config.app_name
    help_line = f"\nQuestions? Write to {config.support_email}." if config.support_email else ""
    return f"\n— {who}{help_line}\n"


def send_verification(to: str, username: str, link: str) -> None:
    send(to, f"Confirm your email for {config.app_name}",
         f"Hello {username},\n\nConfirm this is your email address to start using "
         f"your free daily credits:\n\n{link}\n\nThe link works for 48 hours. If you "
         f"did not create this account, ignore this email.\n" + _sign_off())


def send_password_reset(to: str, username: str, link: str) -> None:
    send(to, f"Reset your {config.app_name} password",
         f"Hello {username},\n\nSomeone (hopefully you) asked to reset the password "
         f"of this account. Choose a new one here:\n\n{link}\n\nThe link works for "
         f"one hour and only once. If you did not ask, ignore this email; your "
         f"password stays as it is.\n" + _sign_off())
