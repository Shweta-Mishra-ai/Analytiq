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
6. **Agree terms with each client in writing.** The licence forbids any use
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
- see each client's storage, the server's memory, disk and running jobs.

There is no self sign-up: every account is one you created.

## Backups

**Download backup** on `/admin` (or `GET /api/admin/backup` with the admin
key) returns one `.tar.gz` of everything: accounts, the signing key and every
client's data. Keep it somewhere other than the server, and treat it as
secret, since it holds all client data.

Daily, from any machine with `curl`:

```
curl -fsS -H "Authorization: Bearer $APP_ADMIN_KEY" \
  https://YOUR-APP/api/admin/backup -o analytiq-$(date +%F).tar.gz
```

Restore: stop the app, unpack the archive so that its `data/` folder becomes
`DATA_DIR`, start the app. Try a restore once before you need one.

## When a client reports a problem

- An error screen quotes a **reference** (for example `3f9a1c2b7d10`). Search
  the server log for `ref=3f9a1c2b7d10` to find the request, the client and
  the full error.
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
- Alerts when the server is down: point an uptime monitor at `/api/health`.
- More than one server process: the queue, rate limit and login throttle are
  per process, and data is on one disk.
