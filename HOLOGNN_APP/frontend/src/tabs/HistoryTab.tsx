import { useState, useEffect, useCallback } from 'react'
import { getHistory, getHistoryItem, deleteHistoryItem, clearHistory, downloadExport } from '../api'
import type { HistoryItem, HistoryRecord, DdgResponse, ScanResponse, IdrResponse, CompareResponse } from '../types'

type KindFilter = 'all' | 'ddg' | 'scan' | 'idr' | 'compare'

const KIND_LABELS: Record<string, string> = {
  ddg: 'ΔΔG',
  scan: 'Scan',
  idr: 'IDR',
  compare: 'Compare',
}

const FILTER_OPTIONS: { value: KindFilter; label: string }[] = [
  { value: 'all',     label: 'All' },
  { value: 'ddg',     label: 'ΔΔG' },
  { value: 'scan',    label: 'Scan' },
  { value: 'idr',     label: 'IDR' },
  { value: 'compare', label: 'Compare' },
]

function KindChip({ kind }: { kind: string }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 10px',
      borderRadius: 12,
      fontSize: '0.78rem',
      fontWeight: 700,
      background: 'var(--teal-light)',
      color: 'var(--teal-dark)',
      border: '1px solid var(--teal-mid)',
      letterSpacing: '0.03em',
    }}>
      {KIND_LABELS[kind] ?? kind}
    </span>
  )
}

function DemoBadge() {
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: 12,
      fontSize: '0.75rem',
      fontWeight: 700,
      background: 'var(--amber-bg)',
      color: '#92400e',
      border: '1px solid var(--amber-border)',
      marginLeft: 6,
    }}>
      demo
    </span>
  )
}

function KindSummary({ record }: { record: HistoryRecord }) {
  const { kind, response } = record

  if (kind === 'ddg') {
    const r = response as unknown as DdgResponse
    return (
      <div className="stat-chips" style={{ marginBottom: 8 }}>
        <div className="stat-chip">Mutation: <strong>{r.mutation}</strong></div>
        <div className="stat-chip">ΔΔG: <strong>{r.ddg != null ? r.ddg.toFixed(3) : '—'} kcal/mol</strong></div>
        <div className="stat-chip">CI: <strong>{r.ci_low != null ? r.ci_low.toFixed(3) : '—'} … {r.ci_high != null ? r.ci_high.toFixed(3) : '—'}</strong></div>
        <div className="stat-chip">Verdict: <strong>{r.verdict}</strong></div>
      </div>
    )
  }

  if (kind === 'scan') {
    const r = response as unknown as ScanResponse
    const positions = r.positions ?? []
    const rangeStr = positions.length > 0
      ? `${positions[0]}–${positions[positions.length - 1]}`
      : '—'
    return (
      <div className="stat-chips" style={{ marginBottom: 8 }}>
        <div className="stat-chip">Positions: <strong>{positions.length}</strong></div>
        <div className="stat-chip">Range: <strong>{rangeStr}</strong></div>
      </div>
    )
  }

  if (kind === 'idr') {
    const r = response as unknown as IdrResponse
    return (
      <div className="stat-chips" style={{ marginBottom: 8 }}>
        <div className="stat-chip">Length: <strong>{r.length}</strong></div>
        <div className="stat-chip">μ: <strong>{r.mu != null ? r.mu.toFixed(4) : '—'}</strong></div>
        <div className="stat-chip">σ: <strong>{r.sigma != null ? r.sigma.toFixed(4) : '—'}</strong></div>
      </div>
    )
  }

  if (kind === 'compare') {
    const r = response as unknown as CompareResponse
    return (
      <div className="stat-chips" style={{ marginBottom: 8 }}>
        <div className="stat-chip">UniProt: <strong>{r.uniprot_id}</strong></div>
        <div className="stat-chip">Sequence length: <strong>{r.sequence ? r.sequence.length : '—'}</strong></div>
      </div>
    )
  }

  return null
}

interface DetailPanelProps {
  item: HistoryItem
  onClose: () => void
  onDeleted: (id: number) => void
}

function DetailPanel({ item, onClose, onDeleted }: DetailPanelProps) {
  const [record, setRecord] = useState<HistoryRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [rawOpen, setRawOpen] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getHistoryItem(item.id)
      .then((r) => { if (!cancelled) { setRecord(r); setLoading(false) } })
      .catch((err: any) => {
        if (!cancelled) {
          setError(err?.response?.data?.detail ?? err?.message ?? 'Failed to load record')
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [item.id])

  async function handleExport() {
    if (!record) return
    setExporting(true)
    try {
      await downloadExport('json', record.response, `holognn_${record.kind}_${record.id}.json`)
    } catch {
      // silently ignore — download errors are uncommon and non-critical
    } finally {
      setExporting(false)
    }
  }

  async function handleDelete() {
    setDeleting(true)
    try {
      await deleteHistoryItem(item.id)
      onDeleted(item.id)
    } catch {
      setDeleting(false)
    }
  }

  return (
    <div style={{
      background: 'var(--teal-light)',
      border: '1px solid var(--teal-mid)',
      borderRadius: 'var(--radius)',
      padding: '16px 20px',
      marginTop: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <span style={{ fontWeight: 600, color: 'var(--teal-dark)', fontSize: '0.9rem' }}>
          Record #{item.id} — {KIND_LABELS[item.kind] ?? item.kind}
        </span>
        <button className="btn btn-secondary" style={{ padding: '4px 12px', fontSize: '0.82rem' }} onClick={onClose}>
          Close
        </button>
      </div>

      {loading && (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          <span className="spinner" />Loading…
        </div>
      )}

      {error && <div className="error-msg">{error}</div>}

      {record && (
        <>
          <KindSummary record={record} />

          <div className="btn-group">
            <button
              className="btn btn-secondary"
              onClick={handleExport}
              disabled={exporting}
              style={{ fontSize: '0.85rem', padding: '6px 14px' }}
            >
              {exporting && <span className="spinner" />}
              Export JSON
            </button>
            <button
              className="btn btn-danger"
              onClick={handleDelete}
              disabled={deleting}
              style={{ fontSize: '0.85rem', padding: '6px 14px' }}
            >
              {deleting && <span className="spinner" />}
              Delete
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setRawOpen((o) => !o)}
              style={{ fontSize: '0.85rem', padding: '6px 14px' }}
            >
              {rawOpen ? 'Hide' : 'Show'} Raw JSON
            </button>
          </div>

          {rawOpen && (
            <pre style={{
              marginTop: 12,
              background: '#1e293b',
              color: '#e2e8f0',
              borderRadius: 'var(--radius)',
              padding: '14px 16px',
              fontSize: '0.78rem',
              overflowX: 'auto',
              maxHeight: 320,
              overflowY: 'auto',
              lineHeight: 1.5,
            }}>
              {JSON.stringify(record.response, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  )
}

interface RowProps {
  item: HistoryItem
  onDeleted: (id: number) => void
}

function HistoryRow({ item, onDeleted }: RowProps) {
  const [expanded, setExpanded] = useState(false)

  function handleDeleted(id: number) {
    setExpanded(false)
    onDeleted(id)
  }

  return (
    <div style={{
      borderBottom: '1px solid var(--border)',
      padding: '12px 0',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <KindChip kind={item.kind} />
        {item.demo_mode && <DemoBadge />}
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
          {new Date(item.created_at * 1000).toLocaleString()}
        </span>
        <span style={{ flex: 1, fontSize: '0.9rem', color: 'var(--text)', minWidth: 120 }}>
          {item.summary}
        </span>
        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
          <button
            className="btn btn-secondary"
            style={{ padding: '4px 12px', fontSize: '0.82rem' }}
            onClick={() => setExpanded((o) => !o)}
          >
            {expanded ? 'Hide' : 'View'}
          </button>
        </div>
      </div>

      {expanded && (
        <DetailPanel item={item} onClose={() => setExpanded(false)} onDeleted={handleDeleted} />
      )}
    </div>
  )
}

export default function HistoryTab() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<KindFilter>('all')

  const fetchHistory = useCallback((kind: KindFilter) => {
    setLoading(true)
    setError(null)
    getHistory(kind === 'all' ? undefined : kind)
      .then((data) => { setItems(data); setLoading(false) })
      .catch((err: any) => {
        setError(err?.response?.data?.detail ?? err?.message ?? 'Failed to load history')
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    fetchHistory(filter)
  }, [filter, fetchHistory])

  function handleDeleted(id: number) {
    setItems((prev) => prev.filter((it) => it.id !== id))
  }

  async function handleClearAll() {
    if (!window.confirm('Clear all prediction history? This cannot be undone.')) return
    try {
      await clearHistory()
      setItems([])
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err?.message ?? 'Clear failed')
    }
  }

  return (
    <div>
      <div className="card">
        <div className="card-title">Prediction History</div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 16, lineHeight: 1.7 }}>
          Every prediction you run is saved to the local database. Browse, inspect, export, or delete records here.
        </p>

        {/* Filter row */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 16 }}>
          {FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`btn ${filter === opt.value ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '5px 14px', fontSize: '0.85rem' }}
              onClick={() => setFilter(opt.value)}
            >
              {opt.label}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          <button
            className="btn btn-secondary"
            style={{ padding: '5px 14px', fontSize: '0.85rem' }}
            onClick={() => fetchHistory(filter)}
            disabled={loading}
          >
            {loading && <span className="spinner" />}
            Refresh
          </button>
          <button
            className="btn btn-danger"
            style={{ padding: '5px 14px', fontSize: '0.85rem' }}
            onClick={handleClearAll}
            disabled={loading || items.length === 0}
          >
            Clear all
          </button>
        </div>

        {error && <div className="error-msg">{error}</div>}

        {loading && !error && (
          <div className="loading-msg">
            <span className="spinner" />Loading history…
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <div style={{
            background: 'var(--teal-light)',
            border: '1px solid var(--teal-mid)',
            borderRadius: 'var(--radius)',
            padding: '14px 18px',
            color: 'var(--teal-dark)',
            fontSize: '0.88rem',
          }}>
            No predictions yet — run one in another tab.
          </div>
        )}

        {!loading && items.length > 0 && (
          <div>
            {items.map((item) => (
              <HistoryRow key={item.id} item={item} onDeleted={handleDeleted} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
