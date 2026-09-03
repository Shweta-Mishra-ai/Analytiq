/**
 * The last step before a number or a column name reaches the screen.
 *
 * This is the TypeScript half of the backend's `present.py`, and it
 * exists because the two halves had drifted: the generated report said
 * "Monthly Income", "8,024" and "Overtime: Yes" while the app beside it
 * said "MonthlyIncome", "8,582.81" and "OverTime: false". The same
 * dataset, described two different ways, by one product.
 */

/** Acronyms a title-caser would otherwise mangle into "Mrr" or "Id". */
const ACRONYMS = new Set([
  'id', 'kpi', 'mrr', 'arr', 'arpu', 'roi', 'roas', 'ctr', 'cpc', 'cpa',
  'cpm', 'ltv', 'cac', 'aov', 'sla', 'nps', 'csat', 'ebitda', 'hr', 'it',
  'fte', 'ytd', 'mtd', 'qtd', 'yoy', 'mom', 'usd', 'eur', 'gbp', 'inr',
  'sku', 'url', 'api', 'ui', 'ux', 'b2b', 'b2c', 'saas', 'gmv', 'aht',
  'otd', 'oee', 'wip', 'bmi', 'los', 'icu', 'er', 'auc', 'vif', 'ci',
])

/** Names the client writes as one word and reads as one word. */
const COMPOUNDS: Record<string, string> = {
  overtime: 'Overtime', headcount: 'Headcount', worklife: 'Work-Life',
  workload: 'Workload', timestamp: 'Timestamp', username: 'Username',
  healthcare: 'Healthcare', onboarding: 'Onboarding', upsell: 'Upsell',
  downtime: 'Downtime', throughput: 'Throughput', churn: 'Churn',
}

const SMALL = new Set(['at', 'of', 'in', 'on', 'per', 'by', 'to', 'for',
  'and', 'or'])

/**
 * A column name as a person would write it.
 * `MonthlyIncome` becomes `Monthly Income`; `years_at_company` becomes
 * `Years at Company`; `mrr_usd` becomes `MRR USD`.
 */
export function label(name: unknown): string {
  const raw = String(name ?? '').trim().replace(/^['"]|['"]$/g, '')
  if (!raw) return ''

  const flat = raw.toLowerCase().replace(/[^a-z0-9]/g, '')
  if (COMPOUNDS[flat]) return COMPOUNDS[flat]
  if (ACRONYMS.has(flat)) return flat.toUpperCase()

  // camelCase and PascalCase split, but never inside a run like B2B or
  // Q4 — a letter/digit boundary is part of the token, not a word break.
  const spaced = raw
    .replace(
      /(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Z][a-z])/g,
      ' ',
    )
    .replace(/[_\-.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  return spaced
    .split(' ')
    .map((word, i) => {
      const low = word.toLowerCase()
      if (ACRONYMS.has(low)) return low.toUpperCase()
      if (i > 0 && SMALL.has(low)) return low
      if (word === word.toUpperCase() && word.length > 1) return word
      return word.charAt(0).toUpperCase() + word.slice(1)
    })
    .join(' ')
}

/**
 * A number sized for reading, never in scientific notation. Precision
 * follows magnitude: a salary wants no decimals, a 1-5 rating wants two.
 */
export function num(value: unknown, opts: { decimals?: number } = {}): string {
  const x = typeof value === 'number' ? value : Number(value)
  if (value === null || value === undefined || !Number.isFinite(x)) return '—'

  if (opts.decimals !== undefined) {
    return x.toLocaleString(undefined, {
      minimumFractionDigits: opts.decimals,
      maximumFractionDigits: opts.decimals,
    })
  }
  const abs = Math.abs(x)
  if (abs >= 1_000_000_000) return (x / 1_000_000_000).toFixed(2) + 'bn'
  if (abs >= 1_000_000) return (x / 1_000_000).toFixed(2) + 'm'
  if (abs >= 100) return Math.round(x).toLocaleString()
  if (abs >= 10) return x.toFixed(1)
  if (abs >= 1) return x.toFixed(2)
  if (x === 0) return '0'
  // Below 1 the useful precision depends on how small it is; a rate of
  // 0.0004 must not print as "0.00".
  const places = Math.min(6, Math.max(2, 1 - Math.floor(Math.log10(abs))))
  return x.toFixed(places).replace(/0+$/, '').replace(/\.$/, '')
}

/**
 * A whole number of things. `num` scales precision to magnitude, which
 * is right for a measure and wrong for a tally: it rendered two unique
 * records as "2.00".
 */
export function count(value: unknown): string {
  const x = Number(value)
  return Number.isFinite(x) ? Math.round(x).toLocaleString() : '—'
}

/**
 * A percentage already expressed on a 0-100 scale.
 *
 * Named for what it takes, because the Insights page multiplied an
 * already-percentage rate by 100 again and showed a 19.9% attrition rate
 * as "1990.0%" — on the headline tile.
 */
export function pct(value: unknown, decimals = 1): string {
  const x = Number(value)
  if (!Number.isFinite(x)) return '—'
  return `${x.toFixed(Math.abs(x) >= 10 ? Math.min(decimals, 1) : decimals)}%`
}
/**
 * A data value as the person who entered it would recognise it.
 *
 * Cleaning turns a Yes/No column into a boolean, and the table then
 * showed "OverTime: false" — a value that appears nowhere in the
 * uploaded file.
 */
export function value(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  if (typeof v === 'number') return num(v)
  const s = String(v).trim().replace(/^['"]|['"]$/g, '')
  if (s.toLowerCase() === 'true') return 'Yes'
  if (s.toLowerCase() === 'false') return 'No'
  return s
}

/** Currency: whole units, because nobody budgets to the cent. */
export function money(v: unknown, symbol = ''): string {
  const x = Number(v)
  if (!Number.isFinite(x)) return '—'
  if (Math.abs(x) >= 1_000_000) return `${symbol}${(x / 1_000_000).toFixed(2)}m`
  return `${symbol}${Math.round(x).toLocaleString()}`
}
/** `a, b and c`, with an honest count when the list runs long. */
export function joinAnd(items: unknown[], limit = 3): string {
  const vals = items.map(String).filter((s) => s.trim())
  if (!vals.length) return ''
  if (vals.length > limit) {
    const extra = vals.length - limit
    return `${vals.slice(0, limit).join(', ')} and ${extra} ${
      extra === 1 ? 'other' : 'others'
    }`
  }
  if (vals.length === 1) return vals[0]
  return `${vals.slice(0, -1).join(', ')} and ${vals[vals.length - 1]}`
}