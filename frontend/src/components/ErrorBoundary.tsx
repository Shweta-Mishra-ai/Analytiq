import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * Catches an error thrown while a screen draws. Without it React removes
 * the whole app and the user is left with a blank page and no way back.
 */
export default class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('screen error', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="flex h-full items-center justify-center p-8" data-testid="error-screen">
        <div className="max-w-md rounded-xl border border-edge bg-panel p-6 text-center">
          <h1 className="text-base font-bold text-ink">This screen could not be shown</h1>
          <p className="mt-2 text-sm text-mute">
            Something went wrong while drawing it. Your data is safe. Reload the page; if it
            happens again on the same screen, tell support which screen it was.
          </p>
          <p className="mt-3 font-data text-[11px] break-words text-faint">
            {String(this.state.error.message || this.state.error).slice(0, 200)}
          </p>
          <div className="mt-4 flex justify-center gap-2">
            <button
              onClick={() => window.location.reload()}
              className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white"
            >
              Reload
            </button>
            <a
              href="/"
              className="rounded-lg border border-edge px-4 py-2 text-xs font-semibold text-ink2"
            >
              Back to start
            </a>
          </div>
        </div>
      </div>
    )
  }
}
