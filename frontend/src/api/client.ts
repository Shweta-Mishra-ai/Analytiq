const BASE = ''
const TOKEN_KEY = 'analytiq-token'

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}
export function setToken(t: string) {
  if (t) localStorage.setItem(TOKEN_KEY, t)
  else localStorage.removeItem(TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

function onUnauthorized() {
  setToken('')
  window.dispatchEvent(new Event('analytiq-unauthorized'))
}

// FastAPI's validation errors put a list of {loc, msg} objects in
// `detail`; passed to `new Error` as-is the user saw "[object Object]".
function describeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      if (d && typeof d === 'object' && 'msg' in d) {
        const loc = Array.isArray((d as { loc?: unknown[] }).loc)
          ? (d as { loc: unknown[] }).loc.filter((p) => p !== 'body').join('.')
          : ''
        return loc ? `${loc}: ${(d as { msg: string }).msg}` : (d as { msg: string }).msg
      }
      return String(d)
    })
    if (parts.length) return parts.join('; ')
  }
  return detail === undefined || detail === null ? fallback : JSON.stringify(detail)
}

async function errorMessage(res: Response): Promise<string> {
  try {
    return describeDetail((await res.json()).detail, res.statusText)
  } catch {
    return res.statusText
  }
}

/** An HTTP error from the API, with its status for callers that branch on it. */
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    // Only a request that carried a token means the session ended. A 401
    // from the login form is a wrong password, and replacing the server's
    // "Wrong username or password" with "Not authenticated" told the user
    // nothing about what they had done wrong.
    const hadToken = Boolean(getToken())
    const message = await errorMessage(res)
    if (hadToken) onUnauthorized()
    throw new ApiError(401, hadToken ? 'Your session has ended — please sign in again' : message)
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return res.json() as Promise<T>
}

export async function apiGet<T>(path: string): Promise<T> {
  return handle<T>(await fetch(`${BASE}${path}`, { headers: authHeaders() }))
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body ?? {}),
    }),
  )
}

export async function apiDelete<T>(path: string): Promise<T> {
  return handle<T>(
    await fetch(`${BASE}${path}`, { method: 'DELETE', headers: authHeaders() }),
  )
}

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return handle<T>(
    await fetch(`${BASE}${path}`, {
      method: 'POST',
      body: form,
      headers: authHeaders(),
    }),
  )
}

export async function apiBlob(
  path: string,
  method: 'GET' | 'POST' = 'GET',
  body?: unknown,
): Promise<Blob> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers:
      method === 'POST'
        ? { 'Content-Type': 'application/json', ...authHeaders() }
        : authHeaders(),
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  })
  if (res.status === 401) {
    onUnauthorized()
    throw new ApiError(401, 'Not authenticated — please log in')
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return res.blob()
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  // Attached and revoked late: revoking in the same tick as click() can
  // cancel the download in Firefox and Safari, and a detached anchor is
  // not clickable in every browser.
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}

// ── shared types ─────────────────────────────────────────
export interface DatasetMeta {
  dataset_id: string
  filename: string
  size_mb: number
  uploaded_at: number
  rows: number
  cols: number
  warnings: string[]
}

export interface TableData {
  columns: string[]
  dtypes: Record<string, string>
  records: Record<string, unknown>[]
  total_rows: number
  truncated: boolean
}

export interface Field {
  name: string
  kind: 'numeric' | 'categorical' | 'datetime'
  missing_pct: number
  unique: number
  values?: string[]
  min?: number | string
  max?: number | string
}

export interface Filter {
  column: string
  op: string
  value: unknown
}

export interface Kpi {
  label: string
  value: number
  format: 'int' | 'pct' | 'num'
  mean?: number
}
