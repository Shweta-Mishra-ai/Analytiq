/**
 * The API client is the one place every page's error handling passes
 * through. If it swallows a 401 or shows "Bad Request" instead of the
 * server's reason, every page in the app degrades at once.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  apiBlob,
  apiGet,
  apiPost,
  apiUpload,
  downloadBlob,
  getToken,
  setToken,
} from './client'

function mockFetch(res: Partial<Response> & { status: number }) {
  const fn = vi.fn().mockResolvedValue({
    ok: res.status >= 200 && res.status < 300,
    statusText: 'Bad Request',
    json: async () => ({}),
    blob: async () => new Blob(),
    ...res,
  } as Response)
  vi.stubGlobal('fetch', fn)
  return fn
}

beforeEach(() => setToken(''))

describe('auth token', () => {
  it('round-trips through localStorage and clears on empty', () => {
    setToken('abc123')
    expect(getToken()).toBe('abc123')
    setToken('')
    expect(getToken()).toBe('')
  })

  it('sends the token as a bearer header once set', async () => {
    setToken('abc123')
    const fetchMock = mockFetch({ status: 200, json: async () => ({ ok: 1 }) })
    await apiGet('/api/datasets')
    const headers = fetchMock.mock.calls[0][1].headers
    expect(headers.Authorization).toBe('Bearer abc123')
  })

  it('sends no Authorization header when logged out', async () => {
    const fetchMock = mockFetch({ status: 200, json: async () => ({}) })
    await apiGet('/api/datasets')
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined()
  })
})

describe('401 handling', () => {
  it('drops the stale token and announces it so the app can re-prompt', async () => {
    setToken('expired')
    mockFetch({ status: 401 })
    const heard = vi.fn()
    window.addEventListener('analytiq-unauthorized', heard)

    await expect(apiGet('/api/datasets')).rejects.toThrow(/Not authenticated/)
    expect(getToken()).toBe('')
    expect(heard).toHaveBeenCalled()
    window.removeEventListener('analytiq-unauthorized', heard)
  })

  it('handles a 401 on a download too, not just on JSON calls', async () => {
    setToken('expired')
    mockFetch({ status: 401 })
    await expect(apiBlob('/api/reports/x/pdf', 'POST', {})).rejects.toThrow(
      /Not authenticated/,
    )
    expect(getToken()).toBe('')
  })
})

describe('error messages', () => {
  it("surfaces the server's reason rather than the status text", async () => {
    mockFetch({
      status: 422,
      json: async () => ({ detail: "Target 'salary' is not in the dataset" }),
    })
    await expect(apiPost('/api/ml/x/train', {})).rejects.toThrow(
      "Target 'salary' is not in the dataset",
    )
  })

  it('falls back to the status text when the body is not JSON', async () => {
    mockFetch({
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => {
        throw new Error('not json')
      },
    })
    await expect(apiGet('/api/datasets')).rejects.toThrow(
      'Internal Server Error',
    )
  })

  it('reports a failed report build with the reason the server gave', async () => {
    mockFetch({
      status: 500,
      json: async () => ({ detail: 'PDF build failed: no numeric columns' }),
    })
    await expect(apiBlob('/api/reports/x/pdf', 'POST', {})).rejects.toThrow(
      'PDF build failed: no numeric columns',
    )
  })
})

describe('request shapes', () => {
  it('posts JSON with the right content type', async () => {
    const fetchMock = mockFetch({ status: 200, json: async () => ({}) })
    await apiPost('/api/ml/x/train', { target: 'attrition' })
    const [, init] = fetchMock.mock.calls[0]
    expect(init.method).toBe('POST')
    expect(init.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(init.body)).toEqual({ target: 'attrition' })
  })

  it('leaves the content type off an upload so the boundary survives', async () => {
    // Setting Content-Type by hand on a FormData post strips the
    // multipart boundary and the server rejects the file.
    const fetchMock = mockFetch({ status: 200, json: async () => ({}) })
    await apiUpload('/api/datasets/upload', new File(['a,b\n1,2'], 'x.csv'))
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers['Content-Type']).toBeUndefined()
    expect(init.body).toBeInstanceOf(FormData)
  })

  it('sends no body on a GET download', async () => {
    const fetchMock = mockFetch({ status: 200, blob: async () => new Blob() })
    await apiBlob('/api/reports/x/csv')
    expect(fetchMock.mock.calls[0][1].body).toBeUndefined()
  })
})

describe('downloadBlob', () => {
  it('names the file and releases the object URL', () => {
    const create = vi.fn().mockReturnValue('blob:fake')
    const revoke = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL: create, revokeObjectURL: revoke })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})

    downloadBlob(new Blob(['x']), 'analytiq_report.pdf')

    expect(click).toHaveBeenCalled()
    // Leaking these keeps the whole blob in memory for the session.
    expect(revoke).toHaveBeenCalledWith('blob:fake')
  })
})
