// Fails on any known vulnerability of moderate severity or above in the
// frontend's dependencies, as `npm audit --audit-level=moderate` did, except
// advisories listed below. Each exception says why it does not reach the app
// and when it lapses: past that date the advisory fails the build again.
import { execFileSync } from 'node:child_process'

const ALLOWED = {
  // picomatch <= 4.0.7: slow matching (ReDoS) on crafted glob patterns.
  // Reached only through Vite and Vitest, which match the repository's own
  // file patterns while building and testing; picomatch is not in the built
  // app (dist/) and the server is Python. 4.0.7 is the newest release, so
  // there is nothing to upgrade to yet.
  'GHSA-fc5q-39xc-vjpf': { until: '2027-01-31', reason: 'build tools only; no fixed release' },
}

const LEVELS = { info: 0, low: 1, moderate: 2, high: 3, critical: 4 }

let raw
try {
  raw = execFileSync('npm', ['audit', '--json'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
} catch (e) {
  raw = e.stdout // npm audit exits non-zero when it finds anything
}
const report = JSON.parse(raw)
const today = new Date().toISOString().slice(0, 10)
const found = new Map()

for (const [pkg, v] of Object.entries(report.vulnerabilities ?? {})) {
  for (const via of v.via) {
    if (typeof via !== 'object' || LEVELS[via.severity] < LEVELS.moderate) continue
    const id = (via.url ?? '').split('/').pop() || `${pkg}:${via.title}`
    found.set(id, { pkg, severity: via.severity, title: via.title, url: via.url })
  }
}

let failed = false
for (const [id, a] of found) {
  const ok = ALLOWED[id]
  if (ok && today <= ok.until) {
    console.log(`allowed until ${ok.until}: ${id} ${a.pkg} (${a.severity}) - ${ok.reason}`)
  } else {
    failed = true
    console.error(`${ok ? 'EXCEPTION EXPIRED' : 'VULNERABLE'}: ${id} ${a.pkg} (${a.severity}) ${a.title} ${a.url ?? ''}`)
  }
}
for (const id of Object.keys(ALLOWED)) {
  if (!found.has(id)) console.log(`no longer reported, remove the exception: ${id}`)
}
console.log(failed ? 'audit failed' : `audit passed (${found.size} advisories, all with a recorded exception)`)
process.exit(failed ? 1 : 0)
