import type { TableData } from '../api/client'

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
            {data.columns.map((c) => (
              <th
                key={c}
                className="border-b border-edge px-3 py-2 font-semibold whitespace-nowrap text-ink"
              >
                {c}
                <span className="ml-1 font-normal text-mute">
                  {data.dtypes[c]?.startsWith('float') ||
                  data.dtypes[c]?.startsWith('int')
                    ? '#'
                    : data.dtypes[c]?.startsWith('datetime')
                      ? '📅'
                      : ''}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.records.map((row, i) => (
            <tr key={i} className="odd:bg-panel even:bg-panel2/40">
              {data.columns.map((c) => (
                <td
                  key={c}
                  className="max-w-52 truncate border-b border-edge/40 px-3 py-1.5 whitespace-nowrap text-mute"
                >
                  {row[c] === null || row[c] === undefined
                    ? '—'
                    : String(row[c])}
                </td>
              ))}
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
