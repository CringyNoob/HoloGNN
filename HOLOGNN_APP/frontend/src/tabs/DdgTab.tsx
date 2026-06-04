import { useState } from 'react'
// @ts-ignore - react-plotly.js types resolved via @types/react-plotly.js
import Plot from 'react-plotly.js'
import { postDdg, downloadExport } from '../api'
import { setDdg } from '../store'
import type { DdgResponse } from '../types'

const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

export default function DdgTab() {
  const [sequence, setSequence] = useState(UBIQUITIN)
  const [mutation, setMutation] = useState('L8P')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DdgResponse | null>(null)
  const [exporting, setExporting] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await postDdg({ wt_sequence: sequence.trim(), mutation: mutation.trim() })
      setResult(res)
      setDdg(res)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleExportJSON() {
    if (!result) return
    setExporting(true)
    try {
      await downloadExport('json', result, `ddg_${result.mutation}.json`)
    } catch {
      setError('Export failed')
    } finally {
      setExporting(false)
    }
  }

  const isStabilizing = result ? result.stabilizing : false
  const verdictClass = isStabilizing ? 'verdict-stabilizing' : 'verdict-destabilizing'
  const markerColor = isStabilizing ? '#2563eb' : '#dc2626'

  const plotData = result
    ? [
        {
          type: 'scatter',
          x: [result.ddg],
          y: [result.mutation],
          mode: 'markers',
          error_x: {
            type: 'data',
            symmetric: false,
            array: [result.ci_high - result.ddg],
            arrayminus: [result.ddg - result.ci_low],
            color: markerColor,
            thickness: 2.5,
            width: 8,
          },
          marker: { size: 14, color: markerColor, symbol: 'circle' },
          name: result.mutation,
        },
      ]
    : []

  const plotLayout = {
    height: 200,
    margin: { l: 80, r: 30, t: 30, b: 50 },
    xaxis: {
      title: 'ΔΔG (kcal/mol)',
      zeroline: true,
      zerolinecolor: '#94a3b8',
      zerolinewidth: 2,
      gridcolor: '#f1f5f9',
    },
    yaxis: { showticklabels: true },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    shapes: [
      {
        type: 'line',
        x0: 0, x1: 0,
        y0: -0.5, y1: 1.5,
        line: { color: '#64748b', width: 1.5, dash: 'dot' },
      },
    ],
    font: { family: 'Inter, system-ui, sans-serif', size: 12 },
  }

  return (
    <div>
      <div className="card">
        <div className="card-title">ΔΔG Predictor — Single Mutation Stability</div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 14 }}>
            <label htmlFor="ddg-seq">Wild-type Protein Sequence</label>
            <textarea
              id="ddg-seq"
              rows={3}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Paste single-letter amino acid sequence..."
              style={{ fontFamily: 'monospace' }}
            />
          </div>
          <div className="input-row">
            <div className="field">
              <label htmlFor="ddg-mut">Mutation (e.g. L8P)</label>
              <input
                id="ddg-mut"
                type="text"
                value={mutation}
                onChange={(e) => setMutation(e.target.value)}
                placeholder="L8P"
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !sequence.trim() || !mutation.trim()}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Predicting…' : 'Predict ΔΔG'}
            </button>
          </div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <div className="card-title">Prediction Result</div>

          <div className="result-block">
            <div className="result-label">ΔΔG Prediction</div>
            <div className={`result-value ${verdictClass}`}>
              {result.ddg > 0 ? '+' : ''}{result.ddg.toFixed(3)} kcal/mol
            </div>
            <div style={{ marginTop: 6, fontSize: '0.95rem', fontWeight: 600 }} className={verdictClass}>
              {result.verdict}
            </div>
          </div>

          <div className="result-row">
            <div className="result-item">
              <div className="result-label">Mutation</div>
              <div style={{ fontWeight: 700, fontFamily: 'monospace', fontSize: '1.1rem' }}>
                {result.wt_residue}{result.position}{result.mut_residue}
              </div>
            </div>
            <div className="result-item">
              <div className="result-label">95% CI (low)</div>
              <div style={{ fontFamily: 'monospace' }}>{result.ci_low.toFixed(3)}</div>
            </div>
            <div className="result-item">
              <div className="result-label">95% CI (high)</div>
              <div style={{ fontFamily: 'monospace' }}>{result.ci_high.toFixed(3)}</div>
            </div>
            <div className="result-item">
              <div className="result-label">Direction</div>
              <div style={{ fontWeight: 600 }} className={verdictClass}>
                {isStabilizing ? '▲ Stabilizing' : '▼ Destabilizing'}
              </div>
            </div>
          </div>

          <div className="plot-wrap section-gap">
            <Plot
              data={plotData}
              layout={plotLayout}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          </div>

          <div className="btn-group">
            <button
              className="btn btn-secondary"
              onClick={handleExportJSON}
              disabled={exporting}
            >
              {exporting && <span className="spinner" />}
              Export JSON
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
