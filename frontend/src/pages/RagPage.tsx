/**
 * RAG Studio — knowledge bases of documents, tables, images and video.
 * Upload anything → ask questions with citations → generate an executive report.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  FileText,
  FileImage,
  FileVideo,
  Table2,
  Layers,
  Plus,
  Send,
  Trash2,
  UploadCloud,
  Sparkles,
  Download,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import {
  apiBlob,
  apiDelete,
  apiGet,
  apiPost,
  apiUpload,
  downloadBlob,
} from '../api/client'
import { Btn, ErrorBox, PageHeader, Panel, Spinner } from '../components/Ui'

interface Kb {
  kb_id: string
  name: string
  files: number
  chunks: number
}

interface KbDetail {
  kb_id: string
  name: string
  embedder: string
  chunks: number
  files: { filename: string; kind: string; chunks: number }[]
}

interface Source {
  ref: number
  source: string
  locator: string
  excerpt?: string
}

interface QaItem {
  question: string
  answer: string
  sources: Source[]
}

const kindIcon = (kind: string) =>
  kind === 'image' ? FileImage : kind === 'video' ? FileVideo : kind === 'table' ? Table2 : FileText

export default function RagPage() {
  const [kbs, setKbs] = useState<Kb[]>([])
  const [active, setActive] = useState<KbDetail | null>(null)
  const [newName, setNewName] = useState('')
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState('')
  const [qa, setQa] = useState<QaItem[]>([])
  const [asking, setAsking] = useState(false)
  const [report, setReport] = useState('')
  const [reporting, setReporting] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refresh = useCallback(() => {
    apiGet<{ knowledge_bases: Kb[] }>('/api/rag/kb')
      .then((r) => setKbs(r.knowledge_bases))
      .catch((e) => setError(e.message))
  }, [])
  useEffect(refresh, [refresh])

  const open = (kbId: string) => {
    setQa([])
    setReport('')
    apiGet<KbDetail>(`/api/rag/kb/${kbId}`)
      .then(setActive)
      .catch((e) => setError(e.message))
  }

  const create = async () => {
    if (!newName.trim()) return
    try {
      const r = await apiPost<{ kb_id: string }>('/api/rag/kb', { name: newName })
      setNewName('')
      refresh()
      open(r.kb_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const upload = async (files: FileList | null) => {
    if (!files || !active) return
    setUploading(true)
    setError('')
    try {
      for (const f of Array.from(files)) {
        await apiUpload(`/api/rag/kb/${active.kb_id}/files`, f)
      }
      open(active.kb_id)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  const ask = async () => {
    if (!question.trim() || !active || asking) return
    const q = question
    setQuestion('')
    setAsking(true)
    setError('')
    try {
      const r = await apiPost<{ answer: string; sources: Source[] }>(
        `/api/rag/kb/${active.kb_id}/query`,
        { question: q },
      )
      setQa((x) => [...x, { question: q, ...r }])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setAsking(false)
    }
  }

  const makeReport = async () => {
    if (!active) return
    setReporting(true)
    setError('')
    try {
      const r = await apiPost<{ markdown: string }>(
        `/api/rag/kb/${active.kb_id}/report`,
        { title: `${active.name} — Analysis Report` },
      )
      setReport(r.markdown)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setReporting(false)
    }
  }

  const downloadPdf = async () => {
    if (!active) return
    const blob = await apiBlob(`/api/rag/kb/${active.kb_id}/report/pdf`, 'POST', {
      title: `${active.name} — Analysis Report`,
    })
    downloadBlob(blob, 'rag_report.pdf')
  }

  return (
    <div className="flex h-full">
      {/* KB list */}
      <div className="w-64 shrink-0 space-y-3 border-r border-edge p-4">
        <h2 className="flex items-center gap-2 text-sm font-bold text-ink">
          <Layers className="h-4 w-4 text-accent" /> Knowledge bases
        </h2>
        <div className="flex gap-1.5">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()}
            placeholder="New KB name…"
            className="min-w-0 flex-1 rounded-lg border border-edge bg-panel px-2.5 py-1.5 text-xs text-ink placeholder:text-mute focus:border-accent focus:outline-none"
          />
          <button
            onClick={create}
            className="rounded-lg bg-accent px-2 text-white disabled:opacity-40"
            disabled={!newName.trim()}
          >
            <Plus className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-1">
          {kbs.map((kb) => (
            <div
              key={kb.kb_id}
              className={`group flex items-center justify-between rounded-lg px-3 py-2 text-xs ${
                active?.kb_id === kb.kb_id
                  ? 'bg-accent/15 text-accent'
                  : 'text-mute hover:bg-panel2 hover:text-ink'
              }`}
            >
              <button onClick={() => open(kb.kb_id)} className="min-w-0 flex-1 text-left">
                <div className="truncate font-semibold">{kb.name}</div>
                <div className="opacity-70">
                  {kb.files} files · {kb.chunks} chunks
                </div>
              </button>
              <button
                onClick={async () => {
                  await apiDelete(`/api/rag/kb/${kb.kb_id}`)
                  if (active?.kb_id === kb.kb_id) setActive(null)
                  refresh()
                }}
                className="hidden text-mute group-hover:block hover:text-rose"
                title="Delete"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          {!kbs.length && (
            <p className="px-1 text-xs text-mute">Create a knowledge base to start.</p>
          )}
        </div>
      </div>

      {/* Main */}
      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        <PageHeader
          title="RAG Studio"
          subtitle="Upload documents, spreadsheets, images and video — then ask questions or generate a report"
          right={
            active ? (
              <div className="flex gap-2">
                <Btn variant="ghost" onClick={makeReport} disabled={reporting || !active.chunks}>
                  <span className="flex items-center gap-1.5">
                    <Sparkles className="h-4 w-4" /> Generate report
                  </span>
                </Btn>
                {report && (
                  <Btn onClick={downloadPdf}>
                    <span className="flex items-center gap-1.5">
                      <Download className="h-4 w-4" /> PDF
                    </span>
                  </Btn>
                )}
              </div>
            ) : undefined
          }
        />
        {error && (
          <div className="mb-4">
            <ErrorBox message={error} />
          </div>
        )}

        {!active ? (
          <Panel>
            <p className="text-sm text-mute">
              Select or create a knowledge base on the left. Supported: PDF, DOCX, TXT/MD,
              CSV/TSV, images (PNG/JPG/WebP) and video (MP4/MOV/WebM). Images and video are
              analyzed with Gemini vision — charts get read, text gets transcribed, narration
              gets summarized — and become searchable alongside your documents.
            </p>
          </Panel>
        ) : (
          <div className="space-y-5">
            <Panel title={`${active.name} — files`}>
              <div
                onClick={() => fileInput.current?.click()}
                className="mb-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl border-2 border-dashed border-edge py-6 text-sm text-mute transition hover:border-accent/50 hover:text-ink"
              >
                <UploadCloud className="h-5 w-5 text-accent" />
                Add files (PDF · DOCX · CSV · images · video)
                <input
                  ref={fileInput}
                  type="file"
                  multiple
                  className="hidden"
                  accept=".pdf,.docx,.txt,.md,.csv,.tsv,.png,.jpg,.jpeg,.webp,.gif,.mp4,.mov,.webm"
                  onChange={(e) => upload(e.target.files)}
                />
              </div>
              {uploading && <Spinner label="Ingesting… media analysis can take a minute" />}
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {active.files.map((f, i) => {
                  const Icon = kindIcon(f.kind)
                  return (
                    <div key={i} className="flex items-center gap-2 rounded-lg bg-panel2 px-3 py-2 text-xs">
                      <Icon className="h-4 w-4 shrink-0 text-accent" />
                      <span className="truncate text-ink">{f.filename}</span>
                      <span className="ml-auto shrink-0 text-mute">{f.chunks}</span>
                    </div>
                  )
                })}
              </div>
            </Panel>

            {/* Q&A */}
            <Panel title="Ask the knowledge base">
              <div className="mb-3 flex gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && ask()}
                  placeholder="e.g. 'What were the key metrics in the video?'"
                  className="flex-1 rounded-xl border border-edge bg-panel2 px-4 py-2.5 text-sm text-ink placeholder:text-mute focus:border-accent focus:outline-none"
                />
                <button
                  onClick={ask}
                  disabled={asking || !question.trim() || !active.chunks}
                  className="rounded-xl bg-accent px-4 text-white disabled:opacity-40"
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
              {asking && <Spinner label="Searching and answering…" />}
              <div className="space-y-4">
                {qa.map((item, i) => (
                  <div key={i} className="rounded-xl border border-edge bg-panel2 p-4">
                    <div className="text-xs font-semibold text-accent">{item.question}</div>
                    <div className="prose prose-sm prose-invert mt-2 max-w-none text-sm [&_p]:my-1">
                      <ReactMarkdown>{item.answer}</ReactMarkdown>
                    </div>
                    {item.sources.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {item.sources.map((s) => (
                          <span
                            key={s.ref}
                            title={s.excerpt}
                            className="rounded-full border border-edge px-2 py-0.5 text-[10px] text-mute"
                          >
                            [{s.ref}] {s.source} · {s.locator}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Panel>

            {reporting && <Spinner label="Writing the executive report…" />}
            {report && (
              <Panel title="Executive report">
                <div className="prose prose-sm prose-invert max-w-none [&_h1]:text-lg [&_h2]:mt-4 [&_h2]:text-base">
                  <ReactMarkdown>{report}</ReactMarkdown>
                </div>
              </Panel>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
