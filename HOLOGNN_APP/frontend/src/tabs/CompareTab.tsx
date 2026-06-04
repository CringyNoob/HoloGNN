import { useState, useRef, useEffect } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import * as $3Dmol from '3dmol'
import { postCompare } from '../api'
import { setCompare } from '../store'
import type { CompareResponse } from '../types'

/** Map pLDDT (stored in B-factor field) to AlphaFold color scheme */
function plddt2color(bfactor: number): string {
  if (bfactor >= 90) return '#0053D6'
  if (bfactor >= 70) return '#65CBF3'
  if (bfactor >= 50) return '#FFDB13'
  return '#FF7D45'
}

export default function CompareTab() {
  const [uniprotId, setUniprotId] = useState('P69905')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CompareResponse | null>(null)

  const viewerRef = useRef<HTMLDivElement>(null)
  const viewerInstance = useRef<$3Dmol.GLViewer | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const res = await postCompare({ uniprot_id: uniprotId.trim() })
      setResult(res)
      setCompare(res)
    } catch (err: unknown) {
      const axiosErr = err as {
        response?: { status?: number; data?: { detail?: string } }
        message?: string
      }
      if (axiosErr.response?.status === 502) {
        setError(
          'Could not reach AlphaFold / UniProt servers (502 Bad Gateway). ' +
          'Check your internet connection or verify the UniProt accession is valid.'
        )
      } else {
        setError(
          axiosErr.response?.data?.detail ??
          axiosErr.message ??
          'Request failed'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  // Initialize the 3Dmol viewer ONCE, then reuse it for every query so we do
  // not leak a new WebGL canvas / context on each compare.
  useEffect(() => {
    if (!result?.structure_pdb || !viewerRef.current) return

    let viewer = viewerInstance.current
    if (!viewer) {
      viewer = $3Dmol.createViewer(viewerRef.current, {
        backgroundColor: '#111111',
        antialias: true,
      })
      viewerInstance.current = viewer
    } else {
      viewer.clear()
    }

    viewer.addModel(result.structure_pdb, 'pdb')
    viewer.setStyle(
      {},
      {
        cartoon: {
          colorfunc: (atom: $3Dmol.AtomSpec) => plddt2color(atom.b ?? 0),
        },
      }
    )
    viewer.zoomTo()
    viewer.render()
  }, [result?.structure_pdb])

  const paeData = result
    ? [
        {
          type: 'heatmap',
          z: result.pae,
          colorscale: 'Viridis',
          reversescale: true,
          colorbar: {
            title: 'PAE (Å)',
            titleside: 'right',
          },
          hovertemplate:
            'Residue i: %{x}<br>Residue j: %{y}<br>PAE: %{z:.2f} Å<extra></extra>',
        },
      ]
    : []

  const paeLayout = {
    height: 280,
    margin: { l: 50, r: 90, t: 30, b: 50 },
    xaxis: { title: 'Residue j', gridcolor: '#f1f5f9' },
    yaxis: { title: 'Residue i', autorange: 'reversed', gridcolor: '#f1f5f9' },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  const overlayData = result
    ? [
        {
          type: 'scatter',
          x: Array.from({ length: result.holognn_min_ddg.length }, (_, i) => i + 1),
          y: result.holognn_min_ddg,
          mode: 'lines',
          line: { color: '#dc2626', width: 2 },
          name: 'Holo-GNN min ΔΔG',
          yaxis: 'y',
          hovertemplate: 'Pos %{x}<br>Min ΔΔG: %{y:.3f}<extra></extra>',
        },
        {
          type: 'scatter',
          x: Array.from({ length: result.plddt.length }, (_, i) => i + 1),
          y: result.plddt,
          mode: 'lines',
          line: { color: '#2563eb', width: 2, dash: 'dot' },
          name: 'AlphaFold pLDDT',
          yaxis: 'y2',
          hovertemplate: 'Pos %{x}<br>pLDDT: %{y:.1f}<extra></extra>',
        },
      ]
    : []

  const overlayLayout = {
    height: 240,
    margin: { l: 55, r: 70, t: 30, b: 50 },
    xaxis: { title: 'Residue position', gridcolor: '#f1f5f9' },
    yaxis: {
      title: 'Min ΔΔG (kcal/mol)',
      titlefont: { color: '#dc2626' },
      tickfont: { color: '#dc2626' },
      gridcolor: '#f1f5f9',
    },
    yaxis2: {
      title: 'pLDDT',
      titlefont: { color: '#2563eb' },
      tickfont: { color: '#2563eb' },
      overlaying: 'y',
      side: 'right',
      range: [0, 100],
    },
    legend: { orientation: 'h', y: -0.3 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', size: 11 },
  }

  return (
    <div>
      <div className="card">
        <div className="card-title">AlphaFold Compare — Structure Confidence vs Predicted Fragility</div>
        <form onSubmit={handleSubmit}>
          <div className="input-row">
            <div className="field">
              <label htmlFor="af-uniprot">UniProt Accession</label>
              <input
                id="af-uniprot"
                type="text"
                value={uniprotId}
                onChange={(e) => setUniprotId(e.target.value)}
                placeholder="e.g. P69905"
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !uniprotId.trim()}
            >
              {loading && <span className="spinner" />}
              {loading ? 'Fetching…' : 'Compare'}
            </button>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 6 }}>
            Fetches AlphaFold structure + PAE from EBI, then runs Holo-GNN scan on the sequence.
          </div>
        </form>
        {error && <div className="error-msg">{error}</div>}
      </div>

      {result && (
        <>
          {result.pae_downsampled && (
            <div style={{
              background: '#eff6ff',
              border: '1px solid #bfdbfe',
              borderRadius: 'var(--radius)',
              padding: '8px 14px',
              fontSize: '0.82rem',
              color: '#1e40af',
              marginBottom: 12,
            }}>
              Note: PAE matrix was downsampled for display performance.
            </div>
          )}

          <div className="compare-layout">
            {/* Left: 3Dmol viewer */}
            <div>
              <div className="card" style={{ marginBottom: 0 }}>
                <div className="card-title">AlphaFold Structure — colored by pLDDT</div>
                <div className="viewer-container" ref={viewerRef} />
                <div className="viewer-legend">
                  <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#0053D6' }} />
                    pLDDT &gt; 90 (very high)
                  </div>
                  <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#65CBF3' }} />
                    70–90 (confident)
                  </div>
                  <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#FFDB13' }} />
                    50–70 (low)
                  </div>
                  <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#FF7D45' }} />
                    &lt; 50 (very low)
                  </div>
                </div>
              </div>
            </div>

            {/* Right: PAE + overlay */}
            <div>
              <div className="card">
                <div className="card-title">AlphaFold PAE (Å)</div>
                <div className="plot-wrap">
                  <Plot
                    data={paeData}
                    layout={paeLayout}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                </div>
              </div>
              <div className="card">
                <div className="card-title">Holo-GNN min ΔΔG vs AlphaFold pLDDT</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                  Red (left axis): most destabilizing predicted mutation per position.
                  Blue dashed (right): AlphaFold confidence.
                </div>
                <div className="plot-wrap">
                  <Plot
                    data={overlayData}
                    layout={overlayLayout}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                    useResizeHandler
                  />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
