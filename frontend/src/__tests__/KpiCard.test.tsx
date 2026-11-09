import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import KpiCard from '../components/KpiCard'
import type { Kpi } from '../api/client'

const kpi = (k: Partial<Kpi>) => ({ label: 'Revenue', value: 0, format: 'num', ...k }) as Kpi

describe('KpiCard', () => {
  it('abbreviates large figures', () => {
    render(<KpiCard kpi={kpi({ value: 9_300_000 })} />)
    expect(screen.getByText('9.3M')).toBeInTheDocument()
  })

  it('formats percentages and counts', () => {
    render(<KpiCard kpi={kpi({ label: 'Margin', value: 21.8, format: 'pct' })} />)
    expect(screen.getByText('21.8%')).toBeInTheDocument()
  })

  it('shows a rise as good in teal and a fall as bad in rose', () => {
    const { rerender } = render(<KpiCard kpi={kpi({ value: 10, delta: 1.8, delta_good: true })} />)
    expect(screen.getByTitle('Change on the prior period')).toHaveClass('text-teal')
    expect(screen.getByTitle('Change on the prior period')).toHaveTextContent('▲ +1.8%')
    rerender(<KpiCard kpi={kpi({ value: 10, delta: -6.3, delta_good: false })} />)
    expect(screen.getByTitle('Change on the prior period')).toHaveClass('text-rose')
    expect(screen.getByTitle('Change on the prior period')).toHaveTextContent('▼ -6.3%')
  })

  it('labels a percentage-point change as pp', () => {
    render(<KpiCard kpi={kpi({ value: 10, delta: 2, delta_unit: 'pp' })} />)
    expect(screen.getByTitle('Change on the prior period')).toHaveTextContent('+2.0 pp')
  })

  it('shows no change badge without a comparison', () => {
    render(<KpiCard kpi={kpi({ value: 10 })} />)
    expect(screen.queryByTitle('Change on the prior period')).toBeNull()
  })
})
