import { useState, useEffect } from 'react'
import { getHealth } from './api'
import type { HealthResponse } from './types'

import DdgTab from './tabs/DdgTab'
import ScanTab from './tabs/ScanTab'
import IdrTab from './tabs/IdrTab'
import CompareTab from './tabs/CompareTab'
import ExportTab from './tabs/ExportTab'

import './styles.css'

const TABS = [
  { id: 'ddg',     label: 'ΔΔG Predictor' },
  { id: 'scan',    label: 'Stability Landscape' },
  { id: 'idr',     label: 'IDR Ensemble' },
  { id: 'compare', label: 'AlphaFold Compare' },
  { id: 'export',  label: 'Export' },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('ddg')
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => {/* silently ignore — backend may not be up in static preview */})
  }, [])

  return (
    <>
      <header className="app-header">
        <h1>Holo-GNN</h1>
        <p>Protein Stability, Interaction &amp; Expression — Graph Neural Network Predictions</p>
      </header>

      {health?.demo_mode && (
        <div className="demo-banner">
          <span style={{ fontSize: '1.1rem' }}>⚠</span>
          <span>
            <strong>Demo mode</strong> — no trained weights loaded. Predictions use a deterministic
            biophysical heuristic. Place <code>holognn_stability_final.pth</code> in the model folder
            for full Holo-GNN inference.
            {health.load_note && <> &nbsp;({health.load_note})</>}
          </span>
        </div>
      )}

      <nav className="tab-bar" role="tablist" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="tab-content" role="tabpanel">
        {activeTab === 'ddg'     && <DdgTab />}
        {activeTab === 'scan'    && <ScanTab />}
        {activeTab === 'idr'     && <IdrTab />}
        {activeTab === 'compare' && <CompareTab />}
        {activeTab === 'export'  && <ExportTab />}
      </main>
    </>
  )
}
