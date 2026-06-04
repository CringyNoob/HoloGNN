import { useState, useEffect } from 'react'
import { downloadExport } from '../api'
import { getStore, subscribe } from '../store'

type Format = 'csv' | 'json' | 'pdb'

export default function ExportTab() {
  const [, forceUpdate] = useState(0)
  const [busy, setBusy] = useState<Format | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Subscribe to store changes so the component re-renders when results arrive
  useEffect(() => {
    const unsub = subscribe(() => forceUpdate((n) => n + 1))
    return unsub
  }, [])

  const store = getStore()
  const hasDdg = !!store.ddg
  const hasScan = !!store.scan
  const hasIdr = !!store.idr

  async function doExport(format: Format) {
    setBusy(format)
    setError(null)
    setSuccess(null)
    try {
      let data: unknown
      let filename: string
      if (format === 'csv') {
        data = store.scan ?? store.ddg ?? store.idr
        filename = 'holognn_export.csv'
      } else if (format === 'json') {
        data = {
          ddg: store.ddg,
          scan: store.scan,
          idr: store.idr,
          compare: store.compare,
        }
        filename = 'holognn_export.json'
      } else {
        // pdb — needs scan data with sequence
        data = store.scan
        filename = 'holognn_export.pdb'
      }
      await downloadExport(format, data, filename)
      setSuccess(`Downloaded ${filename}`)
    } catch {
      setError('Export failed. Check that the backend is running.')
    } finally {
      setBusy(null)
    }
  }

  const anyResult = hasDdg || hasScan || hasIdr || !!store.compare

  return (
    <div>
      <div className="card">
        <div className="card-title">Export Results</div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 16, lineHeight: 1.7 }}>
          Export your computed results in various formats. Run a prediction in one of the other tabs first,
          then come back here to download. The export calls the backend <code>/api/export</code> endpoint,
          which formats and packages the data server-side.
        </p>

        {!anyResult && (
          <div style={{
            background: 'var(--teal-light)',
            border: '1px solid var(--teal-mid)',
            borderRadius: 'var(--radius)',
            padding: '14px 18px',
            color: 'var(--teal-dark)',
            fontSize: '0.88rem',
            marginBottom: 16,
          }}>
            No results yet. Run a prediction in the ΔΔG, Scan, IDR, or Compare tab first.
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
        {success && (
          <div style={{
            background: '#f0fdf4',
            border: '1px solid #bbf7d0',
            borderRadius: 'var(--radius)',
            padding: '10px 16px',
            color: 'var(--green)',
            fontSize: '0.9rem',
            marginBottom: 12,
          }}>
            {success}
          </div>
        )}
      </div>

      <div className="format-grid">
        {/* CSV */}
        <div className="format-card">
          <h3>CSV</h3>
          <p>
            Comma-separated values. Best for stability scan matrices and per-residue
            data. Open in Excel, Python, or R for downstream analysis.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            Uses: {hasScan ? '✓ Scan result' : hasDdg ? '✓ DDG result' : hasIdr ? '✓ IDR result' : '— no data'}
          </div>
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('csv')}
            disabled={!anyResult || busy === 'csv'}
          >
            {busy === 'csv' && <span className="spinner" />}
            Download CSV
          </button>
        </div>

        {/* JSON */}
        <div className="format-card">
          <h3>JSON</h3>
          <p>
            Full structured JSON containing all computed results (ΔΔG, scan, IDR, compare).
            Ideal for programmatic use or archiving a complete analysis session.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            Includes:{' '}
            {[
              hasDdg && 'DDG',
              hasScan && 'Scan',
              hasIdr && 'IDR',
              !!store.compare && 'Compare',
            ].filter(Boolean).join(', ') || '— no data'}
          </div>
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('json')}
            disabled={!anyResult || busy === 'json'}
          >
            {busy === 'json' && <span className="spinner" />}
            Download JSON
          </button>
        </div>

        {/* PDB */}
        <div className="format-card">
          <h3>PDB (B-factor annotated)</h3>
          <p>
            PDB file with Holo-GNN's most-destabilizing ΔΔG per residue encoded in
            the B-factor column. Load in PyMOL or ChimeraX to visualize fragility on
            the 3-D structure.
          </p>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            Requires: {hasScan ? '✓ Scan result (with sequence)' : '— run a scan first'}
          </div>
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={() => doExport('pdb')}
            disabled={!hasScan || busy === 'pdb'}
          >
            {busy === 'pdb' && <span className="spinner" />}
            Download PDB
          </button>
        </div>
      </div>

      {/* Status summary */}
      <div className="card" style={{ marginTop: 8 }}>
        <div className="card-title">Session Data Status</div>
        <div className="stat-chips">
          <div className="stat-chip">
            ΔΔG:{' '}
            {hasDdg
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.ddg!.mutation}</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            Scan:{' '}
            {hasScan
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.scan!.positions.length} positions</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            IDR:{' '}
            {hasIdr
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.idr!.length} residues</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
          <div className="stat-chip">
            Compare:{' '}
            {store.compare
              ? <strong style={{ color: 'var(--green)' }}>✓ {store.compare.uniprot_id}</strong>
              : <span style={{ color: 'var(--text-muted)' }}>not computed</span>}
          </div>
        </div>
      </div>
    </div>
  )
}
