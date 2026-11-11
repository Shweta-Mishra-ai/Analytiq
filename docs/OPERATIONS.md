# Running Analytiq for clients

What to set, what to watch and what to do when something goes wrong. The
figures are measurements of this app, not estimates.

## Before the first client

1. **Memory: at least 2 GB.** The app idles at about 240 MB, reaches 700 MB
   with a few reports running and 1 GB while a 59 MB file loads.
2. **Set `APP_ADMIN_KEY`** to a long random value
   (`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`).
   Without it and with no accounts, the app is open to anyone who has the
   address.
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
- see recent errors and the saved backups.

There is no self sign-up: every account is one you created.

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

The chat and written narratives *inside the web app* are separate: they need
`ANTHROPIC_API_KEY`, `GROQ_API_KEY` or `GEMINI_API_KEY`, and without one the
app shows its calculated findings only.

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
- Billing and invoices.
- Alerts when the server is *down* (as opposed to failing a request): point an
  uptime monitor at `/api/health`.
- Copying backups off the server: schedule the `curl` above somewhere else.
- Self sign-up: every account is created by you.
- More than one server process: the queue, rate limit and login throttle are
  per process, and data is on one disk.
