// Health check
export interface HealthResponse {
  version: string
  model_loaded: boolean
  demo_mode: boolean
  weights_path: string
  load_note: string | null
}

// DDG predictor
export interface DdgRequest {
  wt_sequence: string
  mutation: string
}

export interface DdgResponse {
  mutation: string
  position: number
  wt_residue: string
  mut_residue: string
  ddg: number
  ci_low: number
  ci_high: number
  stabilizing: boolean
  verdict: string
  demo_mode: boolean
}

// Stability scan
export interface ScanRequest {
  sequence: string
  start: number
  end: number
}

export interface ScanResponse {
  positions: number[]
  wt_residues: string[]
  aa_order: string[]
  matrix: number[][]
  demo_mode: boolean
}

// IDR ensemble
export interface IdrRequest {
  sequence: string
}

export interface IdrResponse {
  length: number
  mu: number
  sigma: number
  per_residue: number[]
  demo_mode: boolean
}

// AlphaFold Compare
export interface CompareRequest {
  uniprot_id: string
}

export interface CompareResponse {
  uniprot_id: string
  sequence: string
  structure_pdb: string
  plddt: number[]
  pae: number[][]
  pae_downsampled: boolean
  holognn_min_ddg: number[]
  demo_mode: boolean
}

// Export
export interface ExportRequest {
  format: 'csv' | 'json' | 'pdb'
  filename?: string
  data: unknown
}
