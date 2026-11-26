# Running Analytiq for clients

What to set, what to watch and what to do when something goes wrong. The
figures are measurements of this app, not estimates.

## Before the first client

1. **Memory: at least 2 GB.** The app idles at about 240 MB, reaches 700 MB
   with a few reports running and 1 GB while a 59 MB file loads.
2. **Set `APP_ADMIN_KEY`** to a long random value
   (`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`).
   The Docker image sets `REQUIRE_AUTH=1`, so a deployment without it is
   locked: every request is refused and `/api/health` reports
   `"setup_required": true`. Running from a checkout without Docker, there is
   no such lock: without the key and with no accounts, the app is open to
   anyone who has the address. Set `REQUIRE_AUTH=1` on any such server that
   others can reach.
3. **Serve it over HTTPS.** Passwords and tokens travel in every request.
   Render and Railway do this for you.
4. **Persistent disk** mounted at `DATA_DIR`. Accounts, the token signing key
   and all client data live there.
5. **Set `SUPPORT_EMAIL`** so error screens tell clients whom to contact.
6. **Set `APP_PUBLIC_URL`** to the address clients use
   (`https://analytics.example.com`). The account page shows it as the Claude
   connector address, and report links use it.
7. **Agree terms with each client in writing.** The licence forbids any use
   without written authorization, and you hold their business data.

## Sizing

| Setting | Default | Meaning |
|---|---|---|
| `MAX_FILE_MB` | 50 | Largest upload. Allow about 20 MB of memory per MB of file, on top of 700 MB for the app: 50 MB files fit in 2 GB, 150 MB files need about 4 GB. |
| `MAX_HEAVY_JOBS` | 2 | Uploads, reports and trainings that run at once. Others wait up to `HEAVY_JOB_WAIT` (90 s), then are told the server is busy. With more memory and CPUs, raise it. |
| `MAX_STORAGE_MB` | 1024 | Storage per client. A dataset takes about twice its file size. |
| `MAX_DATASETS` | 50 | Datasets per client. |
| `MIN_FREE_DISK_MB` | 300 | Uploads pause below this much free disk. |
| `RATE_LIMIT_PER_MINUTE` | 300 | API requests per caller per minute. |
| `DATA_TTL_DAYS` | 30 | Days after upload when a dataset is deleted. Clients see the date on the upload page. 0 keeps data until deleted by hand. |

Disk needed ≈ number of clients × `MAX_STORAGE_MB`, plus 1 GB.

## Clients

Open `/admin` on your deployment and enter the admin key. From there:

- create a client (give them the username and password over a channel you
  trust; they can change the password under **Account**);
- reset a lost password (signs out their sessions);
- delete a client and all of their data;
- see each client's storage, the server's memory, disk and running jobs;
- see recent errors and the saved backups;
- see each client's credits, **give credits** by hand (a payment you took
  outside the app) and **make an account unlimited** (never charged).

Anyone can also create their own account at the sign-in screen
(`SIGNUP_ENABLED`, on by default; at most 5 new accounts an hour from one
address). Set `SIGNUP_ENABLED=0` to create every account yourself.

## Credits and payments

Every account gets `FREE_DAILY_CREDITS` (5) free credits a day, renewed at
midnight in `CREDITS_TIMEZONE` (Asia/Kolkata); unused ones do not carry over.
Looking at data, dashboards, charts and insights is free. These cost credits
(`CREDIT_COSTS` changes them, as JSON):

| Action | Key | Credits |
|---|---|---|
| PDF report | `pdf_report` | 2 |
| Data health PDF | `health_report` | 2 |
| Model training | `ml_train` | 1 |
| AI chat answer | `chat` | 1 |
| Knowledge-base report | `kb_report` | 1 |

The Claude connector's report and training tools are charged the same way.
A charge is taken when the action starts and returned automatically if it
fails. Admins, the single-user mode, accounts you mark unlimited on `/admin`
and usernames listed in `FREE_USERS` (e.g. your own) are never charged.

Bought credits never expire. Packs (`CREDIT_PACKS`, JSON; prices in paise
and cents): Starter 50 credits ₹199 / $5, Pro 150 ₹499 / $12, Business 600
₹1,499 / $35.

### Switching on a payment provider

Each one is on as soon as its keys are set; set none, and the Billing page
says to contact support. **Use each provider's test mode first**, buy a pack
end to end, then swap in the live keys.

| Provider | Settings | Webhook to add in its dashboard |
|---|---|---|
| Razorpay: UPI, cards, netbanking, wallets (INR) | `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET` | `APP_PUBLIC_URL/api/billing/webhook/razorpay`, events `order.paid` and `payment.captured` |
| Stripe Checkout | `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_CURRENCY` (`usd` or `inr`) | `APP_PUBLIC_URL/api/billing/webhook/stripe`, events `checkout.session.completed` and `checkout.session.async_payment_succeeded` |
| PayPal (USD) | `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYPAL_MODE` (`sandbox` or `live`) | none: the server captures each payment itself |

Webhooks are a second path: credits are added when the buyer returns to the
app, and when the buyer pays but closes the tab, the next visit to Billing
checks the order with the provider. Set `APP_PUBLIC_URL`: Stripe and PayPal
send buyers back to it.

UPI is taken through Razorpay, which confirms each payment. A bare UPI ID or
QR code gives the app no way to know a transfer arrived; if you accept one,
check it yourself and use **Give credits** on `/admin`.

### Before providers will switch your account on

They check your site for these public pages, which the app serves:
`/pricing`, `/terms`, `/privacy`, `/refunds`, `/contact`. Set
`BUSINESS_NAME`, `BUSINESS_ADDRESS`, `BUSINESS_PHONE` and `SUPPORT_EMAIL`;
until you do, those pages show "not set" where they belong. The policy text
is a starting draft: have it reviewed for your business (refund window,
governing law, GST on invoices).

## Clients who use Claude

No AI key is needed on the server for this. A client with their own Claude
account (any plan; the free plan allows one custom connector) connects it to
Analytiq and asks questions in plain words; Analytiq does the calculations
and Claude writes the answer.

Each client does this once, from **Account → Use Analytiq in Claude**:

1. **Create key.** It is shown once.
2. In Claude: **Customize → Connectors → Add → Add custom connector**.
3. Paste the connector address, choose **No sign in**, and add the request
   header `Authorization` with the value `Bearer <key>`.

The key does not expire, opens the analysis tools only (not the web app or
the password), and stops working when the client replaces or removes it or
you delete the account. Details and the list of tools: [MCP.md](MCP.md).

The AI features *inside the web app* are separate: they need
`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` or a local model
(`LOCAL_LLM_URL`). Without one:

- every analysis, chart, forecast, model and PDF report works as usual (none
  of them uses an AI model to calculate);
- the AI Chat page says chat is not switched on and points clients to the
  Claude connector instead;
- knowledge-base answers and reports quote the matching passages of the
  client's documents, word for word with their sources, instead of writing a
  summary.

## Backups

**Scheduled.** A snapshot of everything (accounts, the signing key, every
client's data) is written every `BACKUP_EVERY_HOURS` (24) to
`DATA_DIR/backups`, and the newest `BACKUP_KEEP` (3) are kept. A snapshot is
skipped, with a warning in the log, when it would leave less than
`MIN_FREE_DISK_MB` free. `/admin` lists them, takes one on demand and
downloads any of them.

These cover a deleted dataset, a bad release or a damaged file. **They do not
cover losing the disk**, because they are on it. For that, copy one off the
server on a schedule, from any machine with `curl`:

```
curl -fsS -H "Authorization: Bearer $APP_ADMIN_KEY" \
  https://YOUR-APP/api/admin/backup -o analytiq-$(date +%F).tar.gz
```

Treat every backup as secret: it holds all client data.

Restore: stop the app, unpack the archive so that its `data/` folder becomes
`DATA_DIR`, start the app. Try a restore once before you need one.

## When a client reports a problem

- An error screen quotes a **reference** (for example `3f9a1c2b7d10`).
  `/admin` lists recent errors by reference with the request, the client and
  the error; the server log has the full traceback under `ref=3f9a1c2b7d10`.
- To be told as errors happen, set `ALERT_WEBHOOK_URL` to an incoming webhook
  (Slack, Discord with `/slack` appended, or similar). At most one message
  every five minutes, with a count of the rest.
- "The server is busy": more heavy jobs were waiting than `MAX_HEAVY_JOBS`
  allows. Occasional is normal; frequent means more memory and a higher limit.
- "Too many requests": the caller passed `RATE_LIMIT_PER_MINUTE`.
- "Storage limit" or "the most this account can hold": the client's quota.
  They delete datasets, or you raise the limit.
- "Out of storage space": the disk is nearly full. Enlarge it or lower
  `DATA_TTL_DAYS`.

Requests that fail with a server error or take over 30 seconds are logged
with their reference, path, duration and client.

## Not included

- Sending email (password reset links, expiry reminders).
- Tax invoices (GST): the payment providers can send receipts; invoices are
  not generated by the app.
- Alerts when the server is *down* (as opposed to failing a request): point an
  uptime monitor at `/api/health`.
- Copying backups off the server: schedule the `curl` above somewhere else.
- Email verification at sign-up and password reset by email: an admin
  resets a lost password on `/admin`.
- More than one server process: the queue, rate limit and login throttle are
  per process, and data is on one disk.
