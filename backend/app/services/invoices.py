"""
services/invoices.py — a numbered invoice (or receipt) for every purchase.

Numbers run without gaps within each Indian financial year (April-March):
AQ-2627-00001, AQ-2627-00002, ... (prefix INVOICE_PREFIX). An invoice is
issued once per payment, whatever number of times the payment is confirmed,
and is kept even if the account is later deleted, as tax records must be.

With GSTIN set, a rupee purchase gets a tax invoice that shows the GST
included in the price at GST_RATE; purchases in other currencies and any
purchase without a GSTIN get a receipt. Which GST applies (IGST or
CGST+SGST, an export under LUT) depends on facts this app does not know:
confirm the setup with your CA.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)


def _fy(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    start = d.year if d.month >= 4 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


class InvoiceBook:
    def __init__(self, root: Optional[str] = None):
        self._lock = threading.RLock()
        self.open(root)

    def open(self, root: Optional[str] = None) -> None:
        self.path = os.path.join(root or os.path.join(config.data_dir, "billing"),
                                 "invoices.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock:
            self._data = {"invoices": [], "counters": {}}
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)

    def issue(self, *, user: str, email: str, description: str, credits: int,
              amount: int, currency: str, provider: str, ref: str) -> dict:
        """The invoice for payment `ref`, issued now if it has none yet."""
        with self._lock:
            for inv in self._data["invoices"]:
                if inv["ref"] == ref:
                    return dict(inv)
            now = time.time()
            fy = _fy(now)
            seq = self._data["counters"].get(fy, 0) + 1
            self._data["counters"][fy] = seq
            taxed = bool(config.gstin) and currency.upper() == "INR"
            rate = float(config.gst_rate) if taxed else 0.0
            taxable = round(amount / (1 + rate / 100)) if taxed else amount
            inv = {"number": f"{config.invoice_prefix}-{fy}-{seq:05d}", "issued_at": now,
                   "user": user, "email": email, "description": description,
                   "credits": credits, "amount": amount, "currency": currency.upper(),
                   "provider": provider, "ref": ref, "tax_invoice": taxed,
                   "gst_rate": rate, "taxable": taxable, "gst": amount - taxable,
                   "gstin": config.gstin if taxed else ""}
            self._data["invoices"].append(inv)
            self._save()
            return dict(inv)

    def for_user(self, user: str) -> list[dict]:
        with self._lock:
            return [dict(i) for i in self._data["invoices"] if i["user"] == user][::-1]

    def get(self, number: str) -> Optional[dict]:
        with self._lock:
            for i in self._data["invoices"]:
                if i["number"] == number:
                    return dict(i)
        return None


invoice_book = InvoiceBook()


def _money(amount: int, currency: str) -> str:
    sym = {"INR": "Rs ", "USD": "$"}.get(currency, currency + " ")
    return "{}{:,.2f}".format(sym, amount / 100)


def render_pdf(inv: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape as e

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=inv["number"], author=config.business_name or config.app_name)
    st = getSampleStyleSheet()
    seller = [config.business_name or config.app_name]
    if config.business_address:
        seller += config.business_address.splitlines()
    if inv["gstin"]:
        seller.append("GSTIN: " + inv["gstin"])
    if config.support_email:
        seller.append(config.support_email)
    title = "Tax Invoice" if inv["tax_invoice"] else "Receipt"
    story = [Paragraph(e(title), st["Title"]),
             Table([[Paragraph("<br/>".join(e(x) for x in seller), st["Normal"]),
                     Paragraph("<b>No.</b> {}<br/><b>Date</b> {}<br/><b>Paid by</b> {}"
                               "<br/><b>Payment ref.</b> {}".format(
                                   e(inv["number"]),
                                   datetime.fromtimestamp(inv["issued_at"]).strftime("%d %b %Y"),
                                   e({"upi": "UPI", "paypal": "PayPal"}.get(
                                       inv["provider"], inv["provider"].title())),
                                   e(inv["ref"].split(":", 1)[-1])),
                               st["Normal"])]], colWidths=[100 * mm, 74 * mm]),
             Spacer(1, 6 * mm),
             Paragraph("<b>Billed to</b><br/>{}{}".format(
                 e(inv["user"]), "<br/>" + e(inv["email"]) if inv["email"] else ""), st["Normal"]),
             Spacer(1, 6 * mm)]
    item = e(inv["description"])
    if config.invoice_sac and inv["tax_invoice"]:
        item += "<br/>SAC " + e(config.invoice_sac)
    rows = [["Description", "Qty", "Amount"],
            [Paragraph(item, st["Normal"]), "1", _money(inv["taxable"], inv["currency"])]]
    if inv["tax_invoice"]:
        rows.append(["GST @ {:g}%".format(inv["gst_rate"]), "", _money(inv["gst"], inv["currency"])])
    rows.append(["Total paid", "", _money(inv["amount"], inv["currency"])])
    t = Table(rows, colWidths=[110 * mm, 20 * mm, 44 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1e3a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [t, Spacer(1, 8 * mm)]
    note = ("Amounts include GST." if inv["tax_invoice"] else
            "This receipt confirms payment received.")
    story.append(Paragraph(e(note + " Credits are added to the account above and do "
                             "not expire."), st["Italic"]))
    doc.build(story)
    return buf.getvalue()


def issue_and_send(*, user: str, description: str, credits: int, amount: int,
                   currency: str, provider: str, ref: str) -> dict:
    """Issue the invoice for a completed payment and email it, if mail is
    set up and the account has an address. Mail trouble never undoes the
    payment: the invoice stays downloadable on the Billing page."""
    from app.services.user_store import user_store
    account = user_store.get(user)
    email = account.email if account else ""
    inv = invoice_book.issue(user=user, email=email, description=description,
                             credits=credits, amount=amount, currency=currency,
                             provider=provider, ref=ref)
    from app.services import emails
    if email and emails.enabled():
        try:
            emails.send(email, f"Your {config.app_name} receipt {inv['number']}",
                        f"Hello {user},\n\nThank you for your payment of "
                        f"{_money(amount, currency)}. {credits} credits have been added to "
                        f"your account.\n\nYour {'tax invoice' if inv['tax_invoice'] else 'receipt'} "
                        f"{inv['number']} is attached, and on the Billing page.\n",
                        attachment=(inv["number"] + ".pdf", render_pdf(inv), "application/pdf"))
        except Exception:
            logger.warning("receipt email for %s not sent", inv["number"], exc_info=True)
    return inv
