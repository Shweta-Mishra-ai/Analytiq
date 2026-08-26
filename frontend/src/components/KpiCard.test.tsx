/**
 * A KPI card is the first number a client reads and often the only one
 * they quote. Formatting it wrong — 4.2K where the answer is 4.2%, or a
 * figure with no indication of which column it came from — is the whole
 * failure mode.
 */
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import KpiCard from './KpiCard'
import type { Kpi } from '../api/client'

const kpi = (over: Partial<Kpi> = {}): Kpi => ({
  label: 'Attrition rate',
  value: 16.1,
  format: 'pct',
  ...over,
})

describe('formatting', () => {
  it('marks a percentage as a percentage', () => {
    render(<KpiCard kpi={kpi()} />)
    expect(screen.getByText('16.1%')).toBeInTheDocument()
  })

  it('groups a large count rather than printing it bare', () => {
    render(<KpiCard kpi={kpi({ label: 'Employees', value: 14700, format: 'int' })} />)
    expect(screen.getByText('14,700')).toBeInTheDocument()
  })

  it('abbreviates thousands and millions on a plain number', () => {
    render(<KpiCard kpi={kpi({ label: 'Revenue', value: 2_430_000, format: 'num' })} />)
    expect(screen.getByText('2.4M')).toBeInTheDocument()
  })

  it('does not abbreviate a number below a thousand', () => {
    render(<KpiCard kpi={kpi({ label: 'Orders', value: 842.5, format: 'num' })} />)
    expect(screen.getByText('842.5')).toBeInTheDocument()
  })

  it('abbreviates a large negative without losing the sign', () => {
    render(<KpiCard kpi={kpi({ label: 'Net change', value: -1_500, format: 'num' })} />)
    expect(screen.getByText('-1.5K')).toBeInTheDocument()
  })
})

describe('context', () => {
  it('names the source column so the figure can be checked', () => {
    render(<KpiCard kpi={kpi({ source_column: 'Attrition' })} />)
    expect(screen.getByText('Attrition rate')).toHaveAttribute(
      'title',
      'From Attrition',
    )
  })

  it('shows a published benchmark next to the figure', () => {
    render(<KpiCard kpi={kpi({ benchmark: 'Industry 10–15%' })} />)
    expect(screen.getByText('Industry 10–15%')).toBeInTheDocument()
  })

  it('falls back to the note when there is no benchmark', () => {
    render(<KpiCard kpi={kpi({ note: 'Twelve months to date' })} />)
    expect(screen.getByText('Twelve months to date')).toBeInTheDocument()
  })

  it('shows the mean only when one was computed', () => {
    const { rerender } = render(<KpiCard kpi={kpi()} />)
    expect(screen.queryByText(/^avg/)).toBeNull()
    rerender(<KpiCard kpi={kpi({ mean: 6502.93 })} />)
    expect(screen.getByText('avg 6,502.9')).toBeInTheDocument()
  })
})
