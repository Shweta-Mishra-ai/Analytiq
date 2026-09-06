import { Calendar } from 'lucide-react'
import type { TableData } from '../api/client'
import * as fmt from '../lib/format'

function isNumericCol(dtype: string | undefined): boolean {
  return !!dtype && (dtype.startsWith('float') || dtype.startsWith('int'))
}

export default function DataTable({
  data,
  maxHeight = '24rem',
}: {
  data: TableData
  maxHeight?: string
}) {
  return (
    <div
      className="overflow-auto rounded-xl border border-edge"
      style={{ maxHeight }}
    >
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-panel2">
          <tr>
            {data.columns.map((c) => {
              // `dtypes` only decides alignment and date formatting. A
              // payload without it should give a plain table, not throw
              // through the render and take the page with it — the line
              // below already allowed for a missing entry, just not a
              // missing map.
              const dtype = data.dtypes?.[c]
              const numeric = isNumericCol(dtype)
              const isDate = dtype?.startsWith('datetime')
              return (
                <th
                  key={c}
                  className={`border-b border-edge px-3 py-2 font-semibold whitespace-nowrap text-ink ${
                    numeric ? 'text-right' : ''
                  }`}
                >
                  {/* The column as the reader would write it. The raw
                      identifier belongs in the tooltip, for anyone
                      matching this back to their own file. */}
                  <span title={c}>{fmt.label(c)}</span>
                  {numeric && (
                    <span className="ml-1 font-data font-normal text-mute">
                      #
                    </span>
                  )}
                  {isDate && (
                    <Calendar className="ml-1 inline h-3 w-3 text-mute" />
                  )}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {data.records.map((row, i) => (
            <tr key={i} className="odd:bg-panel even:bg-panel2/40">
              {data.columns.map((c) => {
                const numeric = isNumericCol(data.dtypes?.[c])
                return (
                  <td
                    key={c}
                    className={`max-w-52 truncate border-b border-edge/40 px-3 py-1.5 whitespace-nowrap text-mute ${
                      numeric ? 'font-data text-right' : ''
                    }`}
                  >
                    {/* fmt.value maps a cleaned boolean back to the
                        Yes/No that was uploaded — the table was showing
                        "false" for a column the user typed "No" into. */}
                    {numeric ? fmt.num(row[c]) : fmt.value(row[c])}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {data.truncated && (
        <div className="bg-panel2 px-3 py-1.5 text-[11px] text-mute">
          Showing {data.records.length.toLocaleString()} of{' '}
          {data.total_rows.toLocaleString()} rows
        </div>
      )}
    </div>
  )
}
