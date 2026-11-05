import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, apiGet, apiPost, getToken, setToken } from '../api/client'

function reply(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      statusText: status === 422 ? 'Unprocessable Entity' : 'Error',
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('sends the stored token as a bearer header', async () => {
    setToken('abc')
    const fetchMock = reply(200, { ok: true })
    vi.stubGlobal('fetch', fetchMock)
    await apiGet('/api/x')
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer abc' })
  })

  it('turns FastAPI validation errors into readable text', async () => {
    vi.stubGlobal('fetch', reply(422, {
      detail: [{ loc: ['body', 'title'], msg: 'field required' }],
    }))
    await expect(apiPost('/api/x', {})).rejects.toThrow('title: field required')
  })

  it('keeps a plain string detail as it is', async () => {
    vi.stubGlobal('fetch', reply(422, { detail: 'This file is a program or script' }))
    const err = (await apiGet("/api/x").catch((e: unknown) => e)) as ApiError
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(422)
    expect(err.message).toBe('This file is a program or script')
  })

  it('ends the session on a 401 when a token was sent', async () => {
    setToken('expired')
    const heard = vi.fn()
    window.addEventListener('analytiq-unauthorized', heard)
    vi.stubGlobal('fetch', reply(401, { detail: 'Not authenticated' }))
    await expect(apiGet('/api/x')).rejects.toThrow('session has ended')
    expect(getToken()).toBe('')
    expect(heard).toHaveBeenCalledOnce()
    window.removeEventListener('analytiq-unauthorized', heard)
  })

  it("shows the server's reason on a 401 from the login form", async () => {
    vi.stubGlobal('fetch', reply(401, { detail: 'Wrong username or password' }))
    await expect(apiPost('/api/login', {})).rejects.toThrow('Wrong username or password')
  })
})
