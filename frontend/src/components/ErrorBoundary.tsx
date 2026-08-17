import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'

/**
 * One render exception used to blank the entire application.
 *
 * React unmounts the whole tree when a render throws and nothing catches
 * it: the user's dataset is still on the server, their analysis is still
 * cached, and what they see is a white page with no navigation and no
 * indication that anything is recoverable. A chart handed a column shape
 * it did not expect was enough to do it.
 *
 * The boundary keeps the failure to the region it happened in. Wrapped
 * around the page outlet, a broken page leaves the sidebar and the
 * dataset picker alive, so the way out is one click rather than a reload
 * and a re-upload.
 */
type Props = { children: ReactNode; label?: string }
type State = { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept in the console rather than sent anywhere: this app holds
    // client data and an error message can carry a column name or a
    // value with it.
    console.error('Page failed to render:', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div className="flex min-h-[60vh] items-center justify-center p-6">
        <div className="surface max-w-lg p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose" />
            <div className="min-w-0">
              <h2 className="t-display text-ink">
                {this.props.label ?? 'This page'} could not be displayed
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-mute">
                Nothing has been lost — your dataset and every analysis
                already run are still on the server. Only this view failed
                to draw.
              </p>
              <p className="mt-3 break-words font-mono text-xs text-faint">
                {error.message || String(error)}
              </p>
              <div className="mt-5 flex gap-2">
                <button
                  type="button"
                  className="btn-primary inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium text-white"
                  onClick={() => this.setState({ error: null })}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Try again
                </button>
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-lg border border-edge px-3 py-1.5 text-xs font-medium text-mute hover:bg-panel2 hover:text-ink"
                  onClick={() => {
                    window.location.href = '/'
                  }}
                >
                  Back to datasets
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}
