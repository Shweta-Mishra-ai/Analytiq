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

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    onUnauthorized()
    throw new Error('Not authenticated — please log in')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
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
    throw new Error('Not authenticated — please log in')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail || detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.blob()
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
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
