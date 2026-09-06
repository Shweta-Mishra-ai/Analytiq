import { useEffect, useRef, useState } from 'react'
import { Send, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { apiGet, apiPost, type Field, type TableData } from '../api/client'
import type { Figure } from '../types'
import { useApp } from '../store/app'
import DataTable from '../components/DataTable'
import PlotlyChart from '../components/PlotlyChart'
import { NeedData, PageHeader } from '../components/Ui'
import * as fmt from '../lib/format'

interface Msg {
  role: 'user' | 'assistant'
  text: string
  figure?: Figure | null
  table?: TableData | null
}

/** Questions written against the columns actually in front of the user.
 *
 *  The four suggestions used to be fixed strings — "Show top 10 rows by
 *  the main numeric column", "Which category has the highest average
 *  value?" A person who does not already know their schema cannot tell
 *  what either would do, and someone who does know it is being asked to
 *  translate. An assistant that has read your file should ask about your
 *  file: "Which region has the highest average revenue?"
 */
function suggestionsFor(fields: Field[]): string[] {
  const usable = fields.filter((f) => f.missing_pct < 90)
  const numeric = usable.filter((f) => f.kind === 'numeric')
  const dates = usable.filter((f) => f.kind === 'datetime')
  // A dimension people can act on: more than one group, few enough that
  // the groups mean something. 40,000 order IDs are not a segmentation.
  const dims = usable
    .filter((f) => f.kind === 'categorical' && f.unique > 1 && f.unique <= 20)
    .sort((a, b) => a.unique - b.unique)

  // Taking numeric[0] means taking whichever measure happens to sit
  // leftmost in the file — on a sales extract that is `units`, and the
  // question a director wants asked is about revenue. Column order is
  // not importance.
  const HEADLINE = [
    'revenue', 'sales', 'profit', 'margin', 'income', 'amount', 'value',
    'spend', 'cost', 'total', 'price', 'score', 'rating', 'units',
    'quantity', 'count',
  ]
  const rank = (name: string) => {
    const flat = name.toLowerCase()
    const hit = HEADLINE.findIndex((w) => flat.includes(w))
    return hit === -1 ? HEADLINE.length : hit
  }
  const ranked = [...numeric].sort((a, b) => rank(a.name) - rank(b.name))
  const metric = ranked[0]?.name
  const dim = dims[0]?.name
  const date = dates[0]?.name
  const out: string[] = []

  if (dim && metric)
    out.push(`Which ${fmt.label(dim)} has the highest average ${fmt.label(metric)}?`)
  if (date && metric)
    out.push(`Show ${fmt.label(metric)} over ${fmt.label(date)} as a line chart`)
  if (metric) out.push(`Show the 10 rows with the highest ${fmt.label(metric)}`)
  if (dim && ranked[1])
    out.push(`Compare ${fmt.label(ranked[1].name)} across ${fmt.label(dim)}`)
  if (numeric.length >= 2) out.push('Which columns move together?')

  // Falls back to the generic four only when the file is too thin to
  // write a specific question about.
  return out.length
    ? out.slice(0, 4)
    : [
        'How many rows and columns are in this file?',
        'Which columns have missing values?',
        'Show me the first 10 rows',
        'Summarise what is in this data',
      ]
}

export default function ChatPage() {
  const dataset = useApp((s) => s.dataset)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const bottom = useRef<HTMLDivElement>(null)
  const ds = dataset?.dataset_id

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  useEffect(() => {
    if (!ds) return
    setSuggestions([])
    apiGet<{ fields: Field[] }>(`/api/charts/${ds}/fields`)
      .then((r) => setSuggestions(suggestionsFor(r.fields ?? [])))
      // A failed lookup costs the tailored questions, not the page: the
      // input still works and the user can type their own.
      .catch(() => setSuggestions([]))
  }, [ds])

  if (!dataset) return <NeedData />

  const send = async (text: string) => {
    if (!text.trim() || busy) return
    setInput('')
    // Snapshot before appending the new user turn, so it becomes the
    // conversation history sent to the backend for follow-up questions.
    const priorTurns = messages
      .filter((m) => m.text)
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.text }))
    setMessages((m) => [...m, { role: 'user', text }])
    setBusy(true)
    try {
      const r = await apiPost<{
        text: string
        figure: Figure | null
        table: TableData | null
      }>(`/api/chat/${ds}`, { message: text, history: priorTurns })
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: r.text, figure: r.figure, table: r.table },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: `⚠ ${e instanceof Error ? e.message : e}` },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col p-8">
      <PageHeader
        title="AI Chat"
        subtitle={`Ask a question about ${dataset.filename} in your own words. Answers come from running real analysis on your data, never from guesswork — and nothing you ask can change the file.`}
        right={
          messages.length > 0 ? (
            <button
              onClick={() => setMessages([])}
              className="flex items-center gap-1.5 text-xs text-mute hover:text-rose"
            >
              <Trash2 className="h-4 w-4" /> Clear
            </button>
          ) : undefined
        }
      />

      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <div className="grid gap-2 sm:grid-cols-2">
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-xl border border-edge bg-panel px-4 py-3 text-left text-sm text-mute transition hover:border-accent/50 hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-3xl rounded-2xl px-4 py-3 ${
              m.role === 'user'
                ? 'ml-auto bg-accent/15 text-ink'
                : 'border border-edge bg-panel'
            }`}
          >
            {m.text && (
              <div className="prose prose-sm prose-invert max-w-none text-sm [&_p]:my-1 [&_ul]:my-1">
                <ReactMarkdown>{m.text}</ReactMarkdown>
              </div>
            )}
            {m.figure && (
              <div className="mt-2 h-80">
                <PlotlyChart figure={m.figure} />
              </div>
            )}
            {m.table && (
              <div className="mt-2">
                <DataTable data={m.table} maxHeight="16rem" />
              </div>
            )}
          </div>
        ))}
        {busy && (
          <div className="max-w-3xl rounded-2xl border border-edge bg-panel px-4 py-3 text-sm text-mute">
            Thinking…
          </div>
        )}
        <div ref={bottom} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="mt-2 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. 'Show average salary by department as a bar chart'"
          className="flex-1 rounded-xl border border-edge bg-panel px-4 py-3 text-sm text-ink placeholder:text-mute focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-xl bg-accent px-4 text-white disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  )
}
