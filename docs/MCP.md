# Using Analytiq from Claude (MCP)

Analytiq exposes its analysis as tools that Claude can call through the
[Model Context Protocol](https://modelcontextprotocol.io). Claude sends a short
request ("summarise this dataset"); Analytiq runs the same tested engines as the
web app and returns a compact answer.

**Why this is cheaper and more reliable than pasting data into a chat**

- **Cost.** Claude never reads the rows. A 2,400-row sales file is about
  142,000 characters. A full session (load the file, business summary, one
  query, PDF report) returns about 2,600 characters, roughly 54× less. The
  replies stay about the same size however large the file is. The tool
  definitions add a fixed ~6,500 characters (about 1.5–2K tokens) to each
  conversation.
- **Quality.** The figures are computed rather than estimated by the model.
  Group differences and correlations are tested, families of tests are
  corrected for multiple comparisons, and periods are whole calendar months.
  Claude is instructed to quote the figures as given and to add no claims of
  its own.

## Tools

| Tool | What it returns |
|---|---|
| `list_datasets` | Your datasets and their ids |
| `load_dataset` | Loads a CSV, Excel or JSON file and returns its profile and id. Accepts `path` (local only), an https `url`, or `csv_text` (up to 200 KB) |
| `business_summary` | Scorecard for this period against the last, key messages with evidence, tested findings, risks and next steps |
| `data_quality` | Quality score and grade, blanks, duplicates, outliers, readiness for modelling |
| `query` | `aggregate`, `filter`, `top_n`, `describe_column`, `correlation`, `count_values`; rankings and correlations say when a gap is within chance |
| `find_drivers` | What predicts a yes/no outcome, and whether the prediction beats chance |
| `train_model` | Compares prediction models on held-out rows against the naive baseline |
| `generate_report` | The consulting-style PDF (`analysis` or `health`, English or Hindi): saved to disk locally, a one-hour download link remotely |

## Option 1 — Claude Code plugin (simplest)

In Claude Code, run:

```text
/plugin install analytiq --marketplace Shweta-Mishra-ai/Analytiq
```

On versions before 2.1.275, run `/plugin marketplace add Shweta-Mishra-ai/Analytiq` and then
`/plugin install analytiq@analytiq`.

The plugin adds the Analytiq tools and an `analyze-data` skill. It needs
**Python 3.11 or newer** on your computer (`python3` on the `PATH`).

On first use, the plugin installs its Python dependencies into a private
environment in Claude Code's plugin data folder. This takes about a minute and
uses about 750 MB; later sessions start immediately. If the tools are not ready
the first time, run `/mcp` and reconnect `plugin:analytiq:analytiq` after a
minute. Progress is logged to `install.log` in
`~/.claude/plugins/data/analytiq-analytiq/`.

The plugin leaves out XGBoost (about 700 MB with its GPU library), so
`train_model` compares three models instead of four. Data stays on your
machine, in `~/.analytiq`.

Windows: the plugin starts the server with `python3`. If your Python is only
available as `python` or `py`, use the manual setup below.

## Option 2 — Local, manual setup (Claude Desktop or Claude Code)

Your data stays on your machine. Files are read by path.

1. Install the backend once:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Claude Code**: run this from any directory, replacing the path:

   ```bash
   claude mcp add --transport stdio --env PYTHONPATH=/path/to/Analytiq/backend \
     analytiq -- python -m app.mcp_server
   ```

   **Claude Desktop**: add this to `claude_desktop_config.json`
   (Settings → Developer → Edit Config), then restart Claude Desktop:

   ```json
   {
     "mcpServers": {
       "analytiq": {
         "command": "/path/to/python",
         "args": ["-m", "app.mcp_server"],
         "env": { "PYTHONPATH": "/path/to/Analytiq/backend" }
       }
     }
   }
   ```

   Use the Python that has the requirements installed (`which python`).

3. Ask Claude, for example: *"Load ~/Downloads/sales.csv with Analytiq, give me
   the business summary, then write the analysis report in Hindi."*

Local datasets are stored in `~/.analytiq` unless `DATA_DIR` is set. Reports
are saved to `~/Downloads` unless you name another path.

## Option 3 — Remote (claude.ai on web and mobile, and your team)

The web app serves the same tools at `https://<your-deployment>/mcp/`. Each person
uses their own Analytiq account and sees only their own datasets.

**Requirements**

- Analytiq must be deployed with HTTPS (see the deployment guides in the
  README) and reachable from the public internet. Behind a firewall, allowlist
  [Anthropic's IP addresses](https://platform.claude.com/docs/en/api/ip-addresses).
- Accounts must be enabled: set `APP_ADMIN_KEY` and create one account per
  person (README → *Multi-tenant clients*). Without accounts the endpoint only
  serves callers on the same machine and refuses anything that arrives through
  a proxy.
- Set `APP_PUBLIC_URL` (for example `https://analytiq.example.com`) so report
  download links use your public address.

**Each person's token**

Sign in once to get a token. It is valid for `TOKEN_TTL_DAYS`, 30 days by default:

```bash
curl -X POST https://<your-deployment>/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "you", "password": "your-password"}'
```

**claude.ai**: *Settings → Connectors → Add custom connector*.

- **URL:** `https://<your-deployment>/mcp/`
- **Authentication:** *No sign in*.
- **Request headers:** `Authorization: Bearer <your token>`.

On Team and Enterprise plans an Owner adds the connector once, and members
connect to it individually. Each member must use **their own** token: a shared
token gives everyone the same account.

**Claude Code (remote)**

```bash
claude mcp add --transport http analytiq https://<your-deployment>/mcp/ \
  --header "Authorization: Bearer <your token>"
```

To share it with a project team, use `--scope project` and keep the token out of
the committed `.mcp.json` with `"Authorization": "Bearer ${ANALYTIQ_TOKEN}"`.

**Getting data in remotely.** Claude on the web cannot hand the server a file
from your computer. You have three ways in:

- upload the file in the Analytiq web app, then refer to its id;
- give Claude an https link to the file;
- paste a small table as CSV text.

## Security

- Remote calls are authorised per request by the same token check as the web
  API. Tokens of deleted accounts stop working.
- Remote callers cannot read server files, and links to private, loopback or
  cloud-metadata addresses are refused, as are redirects.
- Report links are signed, expire after one hour, and name a single file.
- Locally the server runs as you, with your file permissions; only add it to a
  Claude you trust with the files you point it at.

## Licence

Analytiq is proprietary software. Anyone may install and run it, including
through the plugin and the MCP server, for their own personal or internal
business purposes; copying, modifying, redistributing or offering it as a
service is not permitted. See [LICENSE](../LICENSE).
