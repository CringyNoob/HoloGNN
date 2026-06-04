import { useState } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { postScan, downloadExport } from '../api'
import { setScan } from '../store'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

interface ScanResult {
  positions: number[]
  wt_residues: string[]
  aa_order: string[]
  matrix: number[][]
  demo_mode: boolean
  sequence?: string
}

export default function ScanTab() {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [start, setStart] = useState(1)
  const [end, setEnd] = useState(Math.min(UBIQUITIN.length, 60))
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [exporting, setExporting] = useState<string | null>(null)

  function onSeqChange(val: string) {
    setSequence(val)
    // Count residues the way the backend does: drop FASTA header lines + whitespace.
    const len = val
      .split('\n')
      .filter((l) => !l.startsWith('>'))
      .join('')
      .replace(/\s/g, '').length
    setEnd(Math.min(len, 60))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await postScan({
        sequence: sequence.trim(),
        start,
        end,
      })
      const withSeq = { ...res, sequence: sequence.trim() }
      setResult(withSeq)
      setScan(withSeq)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleExport(format: 'csv' | 'pdb') {
    if (!result) return
    setExporting(format)
    try {
      const payload = format === 'pdb'
        ? { ...result, sequence: sequence.trim() }
        : result
      await downloadExport(format, payload, `scan_${start}-${end}.${format}`)
    } catch (err: unknown) {
      setError((err as Error)?.message ?? 'Export failed')
    } finally {
      setExporting(null)
    }
  }

  // Build x-axis labels: wt + position (e.g. "M1", "Q2" …)
  const xLabels = result
    ? result.positions.map((pos, i) => `${result.wt_residues[i] ?? '?'}${pos}`)
    : []

  const plotData = result
    ? [
        {
          type: 'heatmap',
          x: xLabels,
          y: result.aa_order,
          z: result.matrix,
          colorscale: 'RdBu',
          zmid: 0,
          colorbar: {
            title: 'ΔΔG (kcal/mol)',
            titleside: 'right',
            len: 0.9,
          },
          hovertemplate:
            '<b>%{y}%{x}</b><br>ΔΔG: %{z:.3f} kcal/mol<extra></extra>',
          reversescale: true,
        },
      ]
    : []

  const nCols = result ? result.positions.length : 0
  const nRows = 20
  const plotHeight = Math.max(300, nRows * 18 + 80)
  const plotWidth = Math.max(400, nCols * 24 + 120)

  const plotLayout = {
    height: plotHeight,
    margin: { l: 45, r: 110, t: 30, b: 90 },
    xaxis: {
      title: 'Position',
      tickangle: -45,
      automargin: true,
      gridcolor: '#f1f5f9',
    },
    yaxis: {
      title: 'Substituted AA',
      automargin: true,
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  return (
    <div>
      <div className="card">
        <div className="card-title">Stability Landscape — Saturation Scan</div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label htmlFor="scan-seq">Protein Sequence</label>
            <textarea
              id="scan-seq"
              rows={3}
              value={sequence}
              onChange={(e) => onSeqChange(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
            />
          </div>
          <div className="input-row">
            <div className="field-sm">
              <label htmlFor="scan-start">Start (1-based)</label>
              <input
                id="scan-start"
                type="number"
                min={1}
                max={sequence.trim().length}
                value={start}
                onChange={(e) => setStart(Number(e.target.value))}
              />
            </div>
            <div className="field-sm">
              <label htmlFor="scan-end">End (inclusive)</label>
              <input
                id="scan-end"
                type="number"
                min={start}
                max={sequence.trim().length}
                value={end}
                onChange={(e) => setEnd(Number(e.target.value))}
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !sequence.trim()}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Scanning…' : 'Run Scan'}
            </button>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
            Scanning {Math.max(0, end - start + 1)} positions × 20 amino acids.
            Large windows (&gt;80) may be slow.
          </div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <div className="card-title">ΔΔG Heatmap (ΔΔG &gt; 0 = stabilizing ▲, &lt; 0 = destabilizing ▼)</div>
          <div style={{ overflowX: 'auto' }} className="plot-wrap">
            <Plot
              data={plotData}
              layout={{ ...plotLayout, width: plotWidth }}
              config={{ displayModeBar: true, responsive: false }}
              style={{ minWidth: `${plotWidth}px` }}
            />
          </div>
          <div className="btn-group">
            <button
              className="btn btn-secondary"
              onClick={() => handleExport('csv')}
              disabled={exporting === 'csv'}
            >
              {exporting === 'csv' && <span className="spinner" />}
              Export CSV
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => handleExport('pdb')}
              disabled={exporting === 'pdb'}
            >
              {exporting === 'pdb' && <span className="spinner" />}
              Export PDB (B-factor)
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
