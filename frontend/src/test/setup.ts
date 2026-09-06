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

// jsdom implements no layout, so it has no scrollIntoView. Any component
// that keeps a view pinned to the bottom — the chat transcript — throws
// on mount without this, which fails every test in the file for a reason
// that has nothing to do with the component.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
