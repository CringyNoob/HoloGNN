import axios from 'axios'
import type {
  HealthResponse,
  DdgRequest,
  DdgResponse,
  ScanRequest,
  ScanResponse,
  IdrRequest,
  IdrResponse,
  CompareRequest,
  CompareResponse,
  ExportRequest,
  HistoryItem,
  HistoryRecord,
} from './types'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export async function getHealth(): Promise<HealthResponse> {
  const res = await client.get<HealthResponse>('/health')
  return res.data
}

export async function postDdg(payload: DdgRequest): Promise<DdgResponse> {
  const res = await client.post<DdgResponse>('/ddg', payload)
  return res.data
}

export async function postScan(payload: ScanRequest): Promise<ScanResponse> {
  const res = await client.post<ScanResponse>('/scan', payload)
  return res.data
}

export async function postIdr(payload: IdrRequest): Promise<IdrResponse> {
  const res = await client.post<IdrResponse>('/idr', payload)
  return res.data
}

export async function postCompare(payload: CompareRequest): Promise<CompareResponse> {
  const res = await client.post<CompareResponse>('/compare', payload)
  return res.data
}

/**
 * POST /api/export, receive a blob, and trigger a browser download.
 */
export async function downloadExport(
  format: 'csv' | 'json' | 'pdb',
  data: unknown,
  filename?: string
): Promise<void> {
  const payload: ExportRequest = { format, data, filename }
  let res
  try {
    res = await client.post('/export', payload, { responseType: 'blob' })
  } catch (err: any) {
    // With responseType 'blob', error bodies arrive as a Blob — read it back
    // so the caller sees the real {detail: ...} message instead of "[object Blob]".
    if (err?.response?.data instanceof Blob) {
      const text = await err.response.data.text()
      try {
        throw new Error(JSON.parse(text).detail || text)
      } catch {
        throw new Error(text || 'Export failed')
      }
    }
    throw err
  }

  // Try to get filename from Content-Disposition header, else fall back
  const disposition: string = res.headers['content-disposition'] ?? ''
  let dlName = filename ?? `holognn_export.${format}`
  const match = disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
  if (match?.[1]) {
    dlName = match[1].replace(/['"]/g, '')
  }

  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = dlName
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export async function getHistory(kind?: string, limit = 50): Promise<HistoryItem[]> {
  const res = await client.get<{ items: HistoryItem[] }>('/history', { params: { kind, limit } })
  return res.data.items
}

export async function getHistoryItem(id: number): Promise<HistoryRecord> {
  const res = await client.get<HistoryRecord>(`/history/${id}`)
  return res.data
}

export async function deleteHistoryItem(id: number): Promise<void> {
  await client.delete(`/history/${id}`)
}

export async function clearHistory(kind?: string): Promise<void> {
  await client.delete('/history', { params: { kind } })
}
