/**
 * `dtypes` decides column alignment and date formatting. It is not the
 * data, and a table missing it should render plainly rather than throw
 * through the render and take the whole page down with it.
 */
import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import { render } from '../test/render'
import DataTable from './DataTable'
import type { TableData } from '../api/client'

const rows: TableData = {
  columns: ['region', 'revenue'],
  dtypes: { region: 'str', revenue: 'float64' },
  records: [{ region: 'North', revenue: 1200.5 }],
  total_rows: 1,
  truncated: false,
}

describe('DataTable', () => {
  it('renders a normal table', () => {
    render(<DataTable data={rows} />)
    expect(screen.getByText('North')).toBeInTheDocument()
  })

  it('still renders when dtypes is missing entirely', () => {
    const { dtypes: _dropped, ...without } = rows
    render(<DataTable data={without as TableData} />)
    expect(screen.getByText('North')).toBeInTheDocument()
    // headers render through the label formatter: region -> Region
    expect(screen.getByText(/^Region$/)).toBeInTheDocument()
  })

  it('still renders when a single column has no dtype', () => {
    render(<DataTable data={{ ...rows, dtypes: { region: 'str' } }} />)
    expect(screen.getByText('North')).toBeInTheDocument()
  })
})
