# Changelog — Holo-GNN model

Everything introduced for the **project** (model, training/eval, and web app) during the 2026-06
overhaul, newest first.
All additions are backward-compatible: the V6 backbone defaults improve *batched training* but leave
single-sequence inference (used by `predict.py` and the web app) numerically unchanged, and every
heavier capability is an **opt-in flag** that defaults to the original V5 behaviour.

Constructor surface (all optional):

```python
HoloGNN(
    # §8.2 modules (opt-in)
    fusion_mode="concat"|"cross_attention", use_ssm=False, num_species=0,
    # V6 backbone (new defaults shown)
    pool="attention"|"mean", graph_mode="per_sample"|"shared",
    mech_feature_dim=3|6, top_k=8,
    # V6 opt-in heads / training knobs
    antisym_head=False, heteroscedastic=False,
    freeze_esm=False, freeze_esm_layers=0,
)
```

---

## Training, evaluation & deployment (2026-06-05)

Make the project trainable/testable on a friend's **NVIDIA RTX 5070 Ti (Blackwell, sm_120) + AMD**
machine, measurable with real metrics, and runnable with one command.

### GPU / hardware

| Change | Why it's better | How to use |
|--------|-----------------|------------|
| **`src/device.py`** (`get_device`, `describe_device`) | One source of truth for device selection, plus a runtime guard that prints the GPU name + compute capability and warns (with the fix) if the installed torch lacks **sm_120** kernels — the classic Blackwell footgun where the GPU errors or silently drops to CPU. Wired into every train/eval/inference script. | Automatic; every script prints `[device] CUDA: … (sm_120 …)` on launch. |
| **`requirements.txt`** torch floor + cu128 note | A plain `pip install torch` can grab a wheel without sm_120. Pinned `torch>=2.7` and documented the Blackwell install. | `pip install torch --index-url https://download.pytorch.org/whl/cu128` then `pip install -r requirements.txt`. |

### Evaluation metrics

| Change | Why it's better | How to use |
|--------|-----------------|------------|
| **`src/metrics.py`** | Shared, reusable scoring: `regression_metrics` (Pearson, Spearman, RMSE, MAE, R²) and `classification_metrics` (**AUROC, AUPRC, F1**, precision, recall, accuracy, MCC). Guards degenerate batches (single-class → NaN, no crash). | `from src.metrics import regression_metrics, classification_metrics, format_report`. |
| **Validation metrics in training** | `train.py` / `train_final.py` previously created a `val_loader` but never used it; now they (and `train_siamese.py`) print held-out **regression** metrics each epoch, and `train_proteomics.py` prints **AUROC/AUPRC/F1** instead of accuracy-only. | Automatic per epoch. |
| **`evaluate.py`** (new) | One "is the model good on this dataset?" tool. Loads a checkpoint, builds the matching dataset, runs a held-out (tail) split, prints the right metrics, and writes `metrics_<task>.json`. Connects the previously-unused **ClinVar** data via a `pathogenicity` task (ΔΔG destabilisation → `sigmoid(-ΔΔG)` pathogenicity score → AUROC/AUPRC/F1). | `python evaluate.py --task {stability,ddg,pathogenicity,proteomics} --weights w.pth --data <path>`. |

### Web app — persistence & launcher

| Change | Why it's better | How to use |
|--------|-----------------|------------|
| **SQLite history** (`backend/db.py`) | Results used to live only in the browser's in-memory store and vanished on refresh. Every prediction (ΔΔG / scan / IDR / compare) is now persisted to a local SQLite file — the right DB for a single-user, download-and-run app (stdlib only, zero config). | Automatic; DB at `backend/holognn_history.db` (gitignored). |
| **History API + tab** | New `/api/history` routes (`GET` list, `GET /{id}`, `DELETE /{id}`, `DELETE`) and a 6th **History** tab to browse, view, re-export, and delete past predictions. The Export tab now re-hydrates from history on load, so it survives a refresh. | Open the **History** tab. |
| **Compare robustness** | `/api/compare` now returns a clear 502 when an AlphaFold entry has no PDB URL (was a generic parse error). | Automatic. |
| **`runapp.py`** (repo root) | Single cross-platform launcher: bootstraps the backend venv, builds the frontend if needed, starts uvicorn, health-checks, and opens the browser. | `python runapp.py` (add `--full` for real inference, `--weights PATH`, `--port`, `--dev`). |

---

## V6 — Architecture improvements (2026-06-05)

| # | Change | Why it's better | How to use |
|---|--------|-----------------|------------|
| 1 | **Masked attention pooling** (`src/pooling.py`) | V5 pooled with `mean` over the **padded** length, diluting the signal with padding tokens and weighting every residue equally. V6 pools only over **real** residues with learned attention weights. Helps every task. | Default (`pool="attention"`); `pool="mean"` for a padding-aware mean / V5 reproduction. |
| 2 | **Per-sample attention graphs + edge features** (`src/utils/graph_builder.py`) | V5 averaged the whole batch into **one shared** contact graph, so every protein saw the same topology, and it discarded the attention weights (binary threshold). V6 builds **each protein its own graph**, keeps the attention weight as a **GATv2 edge feature** (`edge_dim=1`), uses a stable **top-k** neighbourhood, and guarantees **backbone `(i,i+1)`** connectivity. | Default (`graph_mode="per_sample"`, `top_k=8`); `graph_mode="shared"` for V5. |
| 3 | **Antisymmetric-by-construction ΔΔG head** (`StabilityScoreHead`) | V5's `mlp(z_mt − z_wt)` is **not** guaranteed antisymmetric, so reversibility relied on a soft loss penalty. V6 predicts a scalar stability score `s(z)` and sets `ΔΔG(wt→mt) = s(z_mt) − s(z_wt)`, making `ΔΔG(a→b) = −ΔΔG(b→a)` hold **exactly** (verified: `max|fwd+rev| = 0`). Kills the destabilisation bias structurally. | `HoloGNN(antisym_head=True)` (routes the `idr` task). |
| 4 | **Calibrated uncertainty** (`heteroscedastic`, `src/loss.py:gaussian_nll`, `HeteroscedasticAntisymmetricLoss`) | The UI/paper promise ΔΔG "confidence intervals"; V5 had only a heuristic band. V6 predicts `(μ, σ)` and trains with Gaussian-NLL, so `μ ± 1.96σ` is a genuine ~95% interval. | `HoloGNN(heteroscedastic=True)` → `idr` returns `(μ_fwd, μ_rev, logσ²)`; train with `HeteroscedasticAntisymmetricLoss`. |
| 5 | **Expanded mechanistic features** (`src/dataset.py`) | V5 had 3 channels `[mRNA_fold, CAI, charge]` — two of which are zero for protein-only inputs (ClinVar, the UI). V6 adds three **protein-only** descriptors (Kyte-Doolittle hydropathy, side-chain volume, Chou-Fasman helix propensity), reusing the tables in `src/heuristics.py`. | `HoloGNN(mech_feature_dim=6)` + `mechanistic_features_for_protein(seq, L, expanded=True)`; datasets accept `expanded_mech=True`. |
| 6 | **ESM-2 freezing** (backbone) | Fine-tuning all 8M ESM params overfits small datasets. Freezing all/first-N layers cuts overfitting and speeds training. | `HoloGNN(freeze_esm=True)` or `HoloGNN(freeze_esm_layers=N)`. |
| 7 | **Backbone PyG-fallback fix** | When `torch_geometric` is absent V5 **zeroed** the GAT features (a silent bug). Now a learned projection preserves them. | Automatic. |

---

## §8.2 — Future-work modules (2026-06-04)

Optional modules implementing the paper's §8.2 roadmap; all off by default.

| Change | Where | How to use |
|--------|-------|-----------|
| **Cross-attention fusion** of the ESM and mechanistic tracks | `src/sequence_mixers.py` `CrossAttentionFusion` | `HoloGNN(fusion_mode="cross_attention")` |
| **Mamba / Selective-SSM** linear-time residue mixer (real `mamba_ssm` if installed, else pure-PyTorch SSM) | `src/sequence_mixers.py` `SelectiveSSM` | `HoloGNN(use_ssm=True)` |
| **Three-state head** (destabilising / neutral / stabilising, ±0.5 kcal/mol band) | `src/heads.py` `ThreeStateStabilityHead` | `model((wt, mt), task="three_state")` |
| **MFI multi-task head** (auxiliary mean-fluorescence-intensity regression) | `src/heads.py` `MFIHead` | `model(data, task="mfi")` |
| **Single-sequence stability head** (absolute ΔG regression) | `src/heads.py` `StabilityRegressionHead` | `model(data, task="stability")` |
| **Multi-species proteomics** conditioning | `src/heads.py` `ProteomicsHead(num_species>0)` | `HoloGNN(num_species=N)` + `data.species_id` |
| **Siamese ΔΔG trainer** | `train_siamese.py` | exercises the `idr` task + `AntisymmetricLoss` on FireProtDB pairs |

---

## Correctness fixes (2026-06-04 → 06-05)

- **`predict.py`** rewritten — the V5 script crashed (single-DataBatch `idr` call, missing mechanistic
  features); now does the correct Siamese pass with a deterministic biophysical **demo fallback**.
- **`train.py` / `train_final.py`** — were calling `task="idr"` on single sequences without
  mechanistic features (crash); now train the single-sequence `stability` task correctly.
- **`train_proteomics.py`** — called `backbone(ids, mask)` missing the required `mechanistic_features`
  arg (crash); now routes via `task="proteomics"`.
- **ETL (`master_etl_pipeline.py`)** — stopped labelling
  `Conflicting_interpretations_of_pathogenicity` as Pathogenic (poisoned the ClinVar labels).
- **MLM pre-training (`pretrain_uniref.py`)** — fixed the random-token replacement probability
  (`0.10 → 0.50`, i.e. the intended 10 % of masked positions).
- **`src/dataset.py`** — added the shared `mechanistic_features_for_protein()` entry point used by both
  the CLI and the web app.

---

## Tooling (2026-06-04)

- **`make_figures.py`** — single "image maker" consolidating the three previous figure sources; writes
  to `images/`, with skip-protection for committed figures and a cached-stats demo mode.
- **`requirements.txt`**, consolidated **README**, and the **HOLOGNN_APP** web UI (separate app).
