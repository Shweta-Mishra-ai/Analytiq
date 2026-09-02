/**
 * The integrity panel makes a claim a reader can check: run sha256sum on
 * your file and it matches. These tests hold that the claim is shown,
 * that a failed check is not dressed up as a pass, and that a dataset
 * with no integrity record still renders the rest of the page.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import { render } from '../test/render'
import GovernancePage from './GovernancePage'
import { useApp } from '../store/app'
import * as client from '../api/client'
import type { DatasetMeta } from '../api/client'

const meta: DatasetMeta = {
  dataset_id: 'ds1',
  filename: 'hr.csv',
  size_mb: 1.2,
  uploaded_at: 0,
  rows: 1470,
  cols: 35,
  warnings: [],
}

const governance = {
  source_file: 'hr.csv',
  ingested_at: '2026-09-02',
  rows: 1470,
  columns: 35,
  retention_days: 30,
  retention_note: 'Deleted after 30 days.',
  dictionary: [],
  direct_identifiers: [],
  quasi_identifiers: [],
  special_category: [],
  reidentification: null,
  lineage: [],
  obligations: ['No column identifies a person directly.'],
}

const DIGEST = 'a'.repeat(64)

const integrity = {
  record: {
    source_filename: 'hr.csv',
    source_bytes: 5083,
    source_sha256: DIGEST,
    raw_digest: 'b'.repeat(64),
    active_digest: 'c'.repeat(64),
    ingested_at: 1756800000,
  },
  verdict: {
    intact: true,
    verdict: 'intact',
    explanation: 'The uploaded data is byte-for-byte what was received.',
    events: 2,
  },
  audit: [
    { seq: 1, at: 1756800000, event: 'ingest', actor: 'u', detail: { rows: 1470 } },
    { seq: 2, at: 1756800100, event: 'clean', actor: 'u', detail: { mode: 'non-destructive' } },
  ],
  manifest: { pandas: '3.0.5', numpy: '2.4.6' },
}

function mockApi(integrityBody: unknown | Error) {
  vi.spyOn(client, 'apiGet').mockImplementation((path: string) => {
    if (path.endsWith('/governance')) return Promise.resolve(governance)
    if (path.endsWith('/integrity')) {
      return integrityBody instanceof Error
        ? Promise.reject(integrityBody)
        : Promise.resolve(integrityBody)
    }
    return Promise.reject(new Error(`unexpected ${path}`))
  })
}

describe('GovernancePage integrity panel', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    useApp.setState({ dataset: meta })
  })

  it('shows the source digest the reader can verify themselves', async () => {
    mockApi(integrity)
    render(<GovernancePage />)
    expect(await screen.findByText(DIGEST)).toBeInTheDocument()
    expect(screen.getByText(/sha256sum/)).toBeInTheDocument()
    expect(screen.getByText('Verified')).toBeInTheDocument()
  })

  it('lists the chain of custody', async () => {
    mockApi(integrity)
    render(<GovernancePage />)
    expect(await screen.findByText('Ingest')).toBeInTheDocument()
    expect(screen.getByText('Clean')).toBeInTheDocument()
    expect(screen.getByText(/hash of the one before it/)).toBeInTheDocument()
  })

  it('does not dress a failed check up as a pass', async () => {
    mockApi({
      ...integrity,
      verdict: {
        intact: false,
        verdict: 'unaccounted',
        explanation: 'The working copy does not match any recorded change.',
        events: 2,
      },
    })
    render(<GovernancePage />)
    expect(await screen.findByText('Unverified change')).toBeInTheDocument()
    expect(screen.queryByText('Verified')).not.toBeInTheDocument()
  })

  it('separates a broken audit trail from altered data', async () => {
    mockApi({
      ...integrity,
      verdict: {
        intact: false,
        verdict: 'tampered',
        explanation: 'The record of changes has been edited since it was written.',
        events: 2,
      },
    })
    render(<GovernancePage />)
    expect(await screen.findByText('Audit trail broken')).toBeInTheDocument()
  })

  it('still renders governance when a dataset has no integrity record', async () => {
    mockApi(new Error('404 Not Found'))
    render(<GovernancePage />)
    expect(
      await screen.findByText(/No column identifies a person directly/),
    ).toBeInTheDocument()
    expect(screen.queryByText('Data integrity')).not.toBeInTheDocument()
  })
})
