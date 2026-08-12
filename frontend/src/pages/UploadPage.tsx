import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { UploadCloud, Trash2, FileSpreadsheet, ScanText, FileVideo } from 'lucide-react'
import {
  apiDelete,
  apiGet,
  apiUpload,
  type DatasetMeta,
  type TableData,
} from '../api/client'
import { useApp } from '../store/app'
import DataTable from '../components/DataTable'
import { Btn, ErrorBox, PageHeader, Panel, Spinner } from '../components/Ui'

export default function UploadPage() {
  const nav = useNavigate()
  const { dataset, setDataset } = useApp()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState<TableData | null>(null)
  const [recent, setRecent] = useState<DatasetMeta[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [extractingVideo, setExtractingVideo] = useState(false)

  const refresh = useCallback(() => {
    apiGet<{ datasets: DatasetMeta[] }>('/api/datasets')
      .then((r) => setRecent(r.datasets))
      .catch(() => {})
  }, [])
  useEffect(refresh, [refresh])

  useEffect(() => {
    if (dataset)
      apiGet<TableData>(`/api/datasets/${dataset.dataset_id}/preview?rows=50`)
        .then(setPreview)
        .catch(() => setPreview(null))
  }, [dataset])

  const doUpload = async (file: File) => {
    setBusy(true)
    setError('')
    try {
      const r = await apiUpload<{ meta: DatasetMeta; preview: TableData }>(
        '/api/datasets/upload',
        file,
      )
      setDataset(r.meta)
      setPreview(r.preview)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const doExtract = async (file: File) => {
    setExtracting(true)
    setError('')
    try {
      const r = await apiUpload<{ meta: DatasetMeta; preview: TableData }>(
        '/api/datasets/extract-from-image',
        file,
      )
      setDataset(r.meta)
      setPreview(r.preview)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setExtracting(false)
    }
  }

  const doExtractVideo = async (file: File) => {
    setExtractingVideo(true)
    setError('')
    try {
      const r = await apiUpload<{ meta: DatasetMeta; preview: TableData }>(
        '/api/datasets/extract-from-video',
        file,
      )
      setDataset(r.meta)
      setPreview(r.preview)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setExtractingVideo(false)
    }
  }

  return (
    <div className="p-8">
      <PageHeader
        title="Data Upload"
        subtitle="CSV, Excel (multi-sheet) or JSON up to 200 MB — or let AI extract a table from a photo or video"
      />
      {error && (
        <div className="mb-4">
          <ErrorBox message={error} />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <label
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            if (e.dataTransfer.files[0]) doUpload(e.dataTransfer.files[0])
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed py-14 transition ${
            dragOver
              ? 'border-accent bg-accent/10'
              : 'border-edge bg-panel hover:border-accent/50'
          }`}
        >
          <UploadCloud className="h-10 w-10 text-accent" />
          <p className="mt-3 text-sm text-ink">
            Drop your file here or <span className="text-accent">browse</span>
          </p>
          <p className="mt-1 text-xs text-mute">.csv · .xlsx · .xls · .json</p>
          <input
            type="file"
            accept=".csv,.xlsx,.xls,.json"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && doUpload(e.target.files[0])}
          />
        </label>

        <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-edge bg-panel py-14 transition hover:border-teal/60">
          <ScanText className="h-10 w-10 text-teal" />
          <p className="mt-3 px-4 text-center text-sm text-ink">
            Photo or screenshot of a <span className="text-teal">table</span>
          </p>
          <p className="mt-1 px-4 text-center text-xs text-mute">
            AI extracts it into a real, analyzable dataset
          </p>
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.webp"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && doExtract(e.target.files[0])}
          />
        </label>

        <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-edge bg-panel py-14 transition hover:border-violet-400/60">
          <FileVideo className="h-10 w-10 text-violet-400" />
          <p className="mt-3 px-4 text-center text-sm text-ink">
            Video of a <span className="text-violet-400">table or dashboard</span>
          </p>
          <p className="mt-1 px-4 text-center text-xs text-mute">
            AI reads the data across the clip into a real dataset
          </p>
          <input
            type="file"
            accept=".mp4,.mov,.webm,.avi,.mkv"
            className="hidden"
            onChange={(e) =>
              e.target.files?.[0] && doExtractVideo(e.target.files[0])
            }
          />
        </label>
      </div>

      {busy && <Spinner label="Uploading and profiling…" />}
      {extracting && (
        <Spinner label="Reading the table from your image… (10–30s)" />
      )}
      {extractingVideo && (
        <Spinner label="Reading the table from your video… (30s–a few min)" />
      )}

      {dataset && preview && (
        <div className="mt-6 space-y-4">
          <Panel>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold text-ink">{dataset.filename}</div>
                <div className="text-xs text-mute">
                  {dataset.rows.toLocaleString()} rows × {dataset.cols} columns
                  · {dataset.size_mb} MB
                </div>
                {dataset.warnings?.length > 0 && (
                  <div className="mt-1 text-xs text-amber">
                    ⚠ {dataset.warnings.join(' · ')}
                  </div>
                )}
              </div>
              <Btn onClick={() => nav('/quality')}>Check quality →</Btn>
            </div>
          </Panel>
          <DataTable data={preview} />
        </div>
      )}

      {recent.length > 0 && (
        <Panel title="Recent datasets" className="mt-6">
          <div className="space-y-1.5">
            {recent.map((m) => (
              <div
                key={m.dataset_id}
                className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
                  dataset?.dataset_id === m.dataset_id
                    ? 'bg-accent/10 text-accent'
                    : 'bg-panel2 text-mute hover:text-ink'
                }`}
              >
                <button
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  onClick={() => setDataset(m)}
                >
                  <FileSpreadsheet className="h-4 w-4 shrink-0" />
                  <span className="truncate">{m.filename}</span>
                  <span className="shrink-0 text-xs opacity-70">
                    {m.rows.toLocaleString()} × {m.cols}
                  </span>
                </button>
                <button
                  title="Delete dataset"
                  onClick={async () => {
                    await apiDelete(`/api/datasets/${m.dataset_id}`)
                    if (dataset?.dataset_id === m.dataset_id) setDataset(null)
                    refresh()
                  }}
                  className="ml-3 text-mute hover:text-rose"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  )
}
