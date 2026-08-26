import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

beforeEach(() => {
  // The store persists to localStorage, so without this one test's
  // selected dataset is still selected in the next.
  localStorage.clear()
})
