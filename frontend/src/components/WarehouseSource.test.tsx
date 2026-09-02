/**
 * The warehouse panel handles a credential, so the tests that matter are
 * about what it does with one: never render it in the clear, never keep
 * it, and refuse a write before it can reach the database.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { render } from '../test/render'
import WarehouseSource from './WarehouseSource'
import * as client from '../api/client'

const backends = {
  sqlalchemy: true,
  backends: [
    { key: 'postgresql', label: 'PostgreSQL', available: true, url_prefix: 'postgresql+psycopg://', example: '', install: '' },
    { key: 'snowflake', label: 'Snowflake', available: false, url_prefix: 'snowflake://', example: '', install: 'pip install snowflake-sqlalchemy' },
  ],
}

const URL = 'postgresql+psycopg://alice:hunter2@db.acme.com:5432/prod'

describe('WarehouseSource', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(client, 'apiGet').mockResolvedValue(backends)
  })

  it('names the databases this server can reach, and what to install for the rest', async () => {
    render(<WarehouseSource onImported={() => {}} />)
    expect(await screen.findByText('PostgreSQL')).toBeInTheDocument()
    expect(
      screen.getByText(/pip install snowflake-sqlalchemy/),
    ).toBeInTheDocument()
  })

  it('never renders the connection string in the clear', async () => {
    render(<WarehouseSource onImported={() => {}} />)
    const field = await screen.findByLabelText(/connection url/i)
    await userEvent.type(field, URL)
    // A password field, so a shoulder or a screen share does not leak it.
    expect(field).toHaveAttribute('type', 'password')
    expect(document.body.textContent).not.toContain('hunter2')
  })

  it('says the credential is not stored, because that is the question a user has', async () => {
    render(<WarehouseSource onImported={() => {}} />)
    expect(await screen.findByText(/never saved/)).toBeInTheDocument()
  })

  it('reports a refused write in the server’s own words', async () => {
    vi.spyOn(client, 'apiPost').mockRejectedValue(
      new Error('Only SELECT and WITH queries are allowed.'),
    )
    render(<WarehouseSource onImported={() => {}} />)
    await userEvent.type(await screen.findByLabelText(/connection url/i), URL)
    await userEvent.type(screen.getByLabelText(/query/i), 'DROP TABLE employees')
    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(
      await screen.findByText(/Only SELECT and WITH queries are allowed/),
    ).toBeInTheDocument()
  })

  it('hands the imported dataset up so the rest of the app can use it', async () => {
    const meta = { dataset_id: 'ds9', filename: 'employees', rows: 300, cols: 5 }
    vi.spyOn(client, 'apiPost').mockResolvedValue({ meta })
    const onImported = vi.fn()
    render(<WarehouseSource onImported={onImported} />)
    await userEvent.type(await screen.findByLabelText(/connection url/i), URL)
    await userEvent.type(screen.getByLabelText(/query/i), 'SELECT 1')
    await userEvent.click(screen.getByRole('button', { name: /import as dataset/i }))

    await waitFor(() => expect(onImported).toHaveBeenCalledWith(meta))
  })

  it('survives a server that answers the backends call with something else', async () => {
    vi.spyOn(client, 'apiGet').mockResolvedValue({ unexpected: true })
    render(<WarehouseSource onImported={() => {}} />)
    expect(await screen.findByLabelText(/connection url/i)).toBeInTheDocument()
  })
})
