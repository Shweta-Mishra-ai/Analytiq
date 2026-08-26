/**
 * Pages reach for the router (NeedData links to the upload page), so
 * rendering one bare throws before the assertion is ever reached.
 */
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { render as rtlRender } from '@testing-library/react'

export function render(ui: ReactElement, route = '/') {
  return rtlRender(<MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>)
}
