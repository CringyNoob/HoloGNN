# Holo-GNN

**A unified, geometry-aware Graph Neural Network for protein stability, interaction, and expression prediction — from sequence alone, with no MSAs and no crystal structures.**

<p align="center">
  <img src="images/holognn_final_metrics.png" alt="Holo-GNN performance summary" width="85%">
</p>

Holo-GNN bridges the gap between 1-D protein sequences and 3-D structural biology. Instead of treating a protein as flat text, it **dynamically constructs a graph** of the molecule in real time from an ESM-2 language model's attention map — letting the network "see" 3-D geometry **without** expensive multiple-sequence alignments (MSAs) or crystal-structure (PDB) files. On top of that geometric backbone it learns **protein physics** (Gibbs free energy ΔG / ΔΔG) and transfers that knowledge to downstream tasks such as variant pathogenicity and intrinsically-disordered-region (IDR) ensembles.

---

## Table of contents
- [Why Holo-GNN](#why-holo-gnn)
- [How it works](#how-it-works)
- [Results](#results)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Web app (HOLOGNN_APP)](#web-app-holognn_app)
- [Cloud training (Vertex AI)](#cloud-training-vertex-ai)
- [Repository layout](#repository-layout)
- [What changed over time](#what-changed-over-time)
- [Roadmap / future work](#roadmap--future-work)
- [Authors & citation](#authors--citation)
- [License](#license)

---

## Why Holo-GNN

Existing state-of-the-art stability and structure predictors share three weaknesses that Holo-GNN was designed to remove:

1. **MSA dependency.** Tools like AlphaFold and DDMut rely on multiple-sequence alignments, which are slow to build and simply *fail* on orphan or synthetic proteins that have no evolutionary relatives. Holo-GNN replaces the MSA with a pre-trained **ESM-2** language model, inheriting evolutionary "grammar" directly from sequence.
2. **IDR hallucination.** Models trained to predict a single static 3-D structure hallucinate rigid shapes for **intrinsically disordered regions**, misrepresenting their true dynamic ensembles. Holo-GNN predicts a *distribution* (radius-of-gyration μ/σ) rather than one frozen conformation.
3. **Destabilization bias.** Experimental databases are skewed toward destabilizing mutations, so naïve models memorize the bias and guess "destabilizing" for novel variants — violating thermodynamic reversibility. Holo-GNN enforces **ΔΔG(A→B) = −ΔΔG(B→A)** through a Siamese architecture and an antisymmetric loss.

---

## How it works

Holo-GNN is a **dual-track pipeline** with three modules:

### 1. Sequence module (ESM-2)
The amino-acid sequence is tokenized and embedded by **ESM-2 `t6_8M`** (`facebook/esm2_t6_8M_UR50D`, 320-dim). This sidesteps the orphan-protein bottleneck entirely — no MSA required.

### 2. Structural "Holo" backbone
- **Dynamic graph construction.** Edges between residues are drawn from the ESM-2 **attention map**: if residue *A* strongly attends to residue *B*, an edge is created, approximating a physical 3-D contact (`build_attention_graph`, [`src/backbone.py`](src/backbone.py)).
- **Mechanistic feature injection.** Three biophysical channels are concatenated onto every node before message passing: **mRNA-fold proxy** (codon GC-skew × stacking energy), **Codon Adaptation Index** (CAI, *E. coli* K-12 usage), and **local charge** (Henderson–Hasselbalch at pH 7.4). See [`src/dataset.py`](src/dataset.py).
- **GATv2 + residual.** Two **GATv2Conv** layers (Brody et al., ICLR 2022 — strictly more expressive *dynamic* attention than GATv1) refine the graph, with a **residual skip connection** from the raw ESM-2 embeddings plus LayerNorm to prevent over-smoothing.

### 3. Task heads ([`src/heads.py`](src/heads.py))
| Head | `task=` | Output |
|------|---------|--------|
| `SiameseStabilityHead` | `idr` (paired) | ΔΔG, both directions (kcal/mol) |
| `StabilityRegressionHead` | `stability` | single-sequence absolute ΔG |
| `ProteomicsHead` | `proteomics` | retention-time / expression / pathogenicity |
| `EnsembleIDRHead` | *(model.idr_head)* | IDR radius-of-gyration (μ, σ) |
| `ThreeStateStabilityHead` | `three_state` | destabilising / neutral / stabilising (§8.2) |
| `MFIHead` | `mfi` | mean fluorescence intensity (§8.2) |

**Siamese antisymmetry.** For a mutation, the wild-type and mutant are each passed through the backbone independently, and the difference embedding `(z_mt − z_wt)` is scored in **both directions**. The antisymmetric loss penalizes any deviation from `ΔΔG(WT→MT) = −ΔΔG(MT→WT)`, eliminating destabilization bias. See `forward(data, task="idr")` in [`src/full_model.py`](src/full_model.py).

---

## Results

Trained in two phases — **physics pre-training** on stability, then **medical transfer** to pathogenicity.

### Phase 1 — Stability (ΔΔG)
Evaluated on the **Tsuboyama et al. mega-scale** dataset (1,841,285 independent mutations):

| Metric | Value |
|--------|------:|
| Pearson *r* | **0.7644** |
| MAE | **1.6496 kcal/mol** |
| RMSE | **2.0163 kcal/mol** |
| Mean antisymmetry violation | **0.7408 kcal/mol** |

<p align="center">
  <img src="images/Figure_1_Correlation.png" alt="Predicted vs. experimental ΔΔG" width="48%">
  <img src="images/training_loss_curve.png" alt="Training loss curve" width="48%">
</p>

### Phase 2 — Pathogenicity (helix-breaker transfer experiment)
Backbone frozen; only the classification head trained, then validated on 500 synthetic structural variants:

| Metric | Value |
|--------|------:|
| ROC-AUC | **0.84** |
| Specificity | **95.1 %** |
| Accuracy | **68.8 %** |

The model is deliberately **risk-averse** — it almost never flags a benign variant (high specificity), at the cost of sensitivity. Confusion matrix and ROC below:

<p align="center">
  <img src="images/Figure_2_ConfusionMatrix.png" alt="Confusion matrix" width="42%">
  <img src="images/Figure_3_ROC.png" alt="ROC curve" width="42%">
</p>

> All figures are reproducible with `python make_figures.py --figure all` (writes to `images/`).

---

## Datasets

Holo-GNN is trained on ~68 GB of public data spanning four task families. Full per-file inventory: [`DATASET_INVENTORY.md`](DATASET_INVENTORY.md).

| Task | Source | Scale |
|------|--------|------:|
| **Stability** | Tsuboyama 2023 mega-scale cDNA (K50 dG) | ~1.84 M mutations |
| **Stability** | FireProtDB | 1.7 GB experimental ΔΔG |
| **Pre-training** | UniRef50 (FASTA) | ~23 GB sequences |
| **Pathogenicity** | ClinVar VCF | ~1.9 GB variants |
| **Proteomics** | MassIVE-KB spectral library | ~27 GB MS/MS |

Raw data is **not** checked into the repository (see [`.gitignore`](.gitignore)); it is streamed/extracted by the ETL pipeline ([`master_etl_pipeline.py`](master_etl_pipeline.py)).

---

## Installation

**Requirements:** Python 3.9+ (3.10–3.12 recommended). A GPU is optional — everything runs on CPU.

```bash
# 1. Get the code
git clone https://github.com/CringyNoob/HoloGNN.git
cd HoloGNN

# 2. Create an isolated environment (required on modern, PEP-668 systems)
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The **first run downloads the ESM-2 `t6_8M` weights (~30 MB)** from Hugging Face (needs internet once).
- **`torch_geometric` is optional.** If it is missing the backbone bypasses the GATv2 layers via a fallback projection (it no longer zeroes the features) so the model still runs; install it for the full dynamic-attention path.
- The repo ships **no trained weights** (`*.pth` is git-ignored). Without them, `predict.py` runs a deterministic biophysical heuristic ("demo mode"). Provide `holognn_stability_final.pth` (or set `HOLOGNN_WEIGHTS`) for full inference.

---

## Quick start

### Predict a stability change (ΔΔG)
```bash
# Built-in ubiquitin M1A demo:
python predict.py

# Your own protein + point mutation (e.g. Leu-8 → Pro):
python predict.py --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ... --mut L8P

# Two full sequences:
python predict.py --wt <wildtype> --mt <mutant>
```

**No trained weights?** The public repo intentionally ships none (`*.pth` is git-ignored). `predict.py` then runs in **demo mode**, using a deterministic biophysical heuristic ([`src/heuristics.py`](src/heuristics.py)) so the tool is always runnable. Drop `holognn_stability_final.pth` in the project root (or set `HOLOGNN_WEIGHTS`) to switch to full Holo-GNN inference.

> **Sign convention:** ΔΔG > 0 → stabilizing, ΔΔG < 0 → destabilizing (ΔG_mut − ΔG_wt).

### Train
Training needs the datasets locally (see [Datasets](#datasets)); they are produced by `master_etl_pipeline.py`.

```bash
python train_siamese.py      # flagship: Siamese ΔΔG + AntisymmetricLoss (FireProtDB pairs)
python train_final.py        # single-sequence absolute ΔG (MegaScale, gradient-accumulated)
python train_proteomics.py   # transfer-learn the pathogenicity head (backbone frozen)
python pretrain_uniref.py    # optional masked-LM pre-training on UniRef50 shards
```

The model exposes tasks via `model(data, task=...)`: `idr` (paired ΔΔG), `stability`, `proteomics`, `three_state`, `mfi`. See `forward()` in [`src/full_model.py`](src/full_model.py).

### Regenerate all paper figures
```bash
python make_figures.py --figure all          # → images/
python make_figures.py --figure correlation  # individual figures also supported
```

---

## Web app (HOLOGNN_APP)

A browser-based dashboard (paper §8.1) lets non-computational researchers use Holo-GNN with no code. It lives in the sibling [`../HOLOGNN_APP`](../HOLOGNN_APP) folder (React + FastAPI), is meant to be **downloaded and run locally**, and offers:

- **ΔΔG predictor** with confidence intervals,
- an interactive **stability-landscape heatmap** (every substitution × position),
- **IDR ensemble** distribution plots,
- side-by-side **AlphaFold2 pLDDT / PAE** comparison, and
- **CSV / JSON / PDB-annotation** export.

It works against the same model package and falls back to demo mode when no weights are present. See `../HOLOGNN_APP/README.md` for setup.

---

## Cloud training (Vertex AI)

Holo-GNN was scaled on **Google Vertex AI**. Recommended configurations:

| GPU | VRAM | `BATCH_SIZE` | `num_workers` | Notes |
|-----|-----:|-------------:|--------------:|-------|
| **NVIDIA L4** | 24 GB | 64 | 8 | production config; gradient accumulation no longer needed |
| **NVIDIA T4** | 16 GB | 32 | 8 | budget config |
| GTX 1050 Ti (local) | 4 GB | 4 (×8 accum → 32 eff.) | 2 | original prototype hardware |

Typical workflow: `gsutil -m rsync` the dataset bucket → extract to `data/mega_scale_cdna/...` → `pip install torch_geometric transformers biopython` → run `train_final.py` with the GPU-appropriate batch size → evaluate and call `make_figures.py`. Learning rate `1e-4` with cosine annealing; loss = MSE (stability) / antisymmetric (Siamese) / BCE (pathogenicity).

---

## Repository layout

```
HoloGNN/
├── src/
│   ├── full_model.py        # HoloGNN multi-task model (all tasks + §8.2 flags)
│   ├── backbone.py          # ESM-2 → (opt. SSM) → (opt. cross-attn) fusion → GATv2 + residual
│   ├── heads.py             # Siamese/Stability/Proteomics/IDR + ThreeState/MFI heads
│   ├── sequence_mixers.py   # §8.2 CrossAttentionFusion + Mamba/SelectiveSSM
│   ├── dataset.py           # mechanistic features + dataset loaders
│   ├── heuristics.py        # deterministic biophysical demo fallback
│   ├── loss.py              # antisymmetric loss
│   └── utils/graph_builder.py
├── predict.py               # CLI ΔΔG predictor (full + demo modes)
├── make_figures.py          # single "image maker" — regenerates every figure
├── train_siamese.py         # Siamese ΔΔG + AntisymmetricLoss (flagship)
├── train_final.py / train.py / train_proteomics.py / pretrain_uniref.py
├── master_etl_pipeline.py / pipeline.py    # data ETL + orchestration
├── dataset_scanner.py + DATASET_INVENTORY.md
├── images/                  # all figures live here
└── requirements.txt
```

---

## What changed over time

Holo-GNN evolved through five architectures. Each step targeted a specific failure mode:

- **V1 — Linear-graph MVP.** Residues connected only to neighbours (i → i+1); ESM-2 swapped in for MSAs. Proved the concept but scaled poorly (*r* ≈ 0.42 on orphan targets).
- **V2 — Attention graph.** Edges built from the ESM-2 attention map instead of a fixed chain, giving the model real long-range "contacts." Solved the orphan-protein bottleneck.
- **V3 — Mechanistic injection + GAT.** Added the three biophysical channels (CAI, charge, mRNA-fold) and Graph Attention layers; MAE dropped to **2.30 kcal/mol**.
- **V4 — Siamese + antisymmetric loss.** Wild-type and mutant scored in tandem with an antisymmetry constraint, **eliminating destabilization bias** and breaking *r* > 0.70 for the first time.
- **V5 — Production (current).** GATv1 → **GATv2** dynamic attention, residual skip connections, *true* mechanistic features, and full cloud scaling (batch 64) on **1.84 M** MegaScale mutations → **r = 0.7644, MAE = 1.65, RMSE = 2.02 kcal/mol**.

---

## Future-work features (paper §8.2) — implemented & experimental

The §8.2 directions are now wired into the architecture as **optional, off-by-default** modules, so existing behaviour and checkpoints are unchanged unless you opt in:

```python
from src.full_model import HoloGNN

# §8.2-i  Cross-Attention Fusion — cross-modal attention between the ESM and
#          mechanistic tracks instead of plain concatenation.
model = HoloGNN(fusion_mode="cross_attention")

# §8.2-ii Mamba / Selective State-Space mixer — linear-time residue mixing for
#          long sequences (uses real `mamba_ssm` if installed, else a pure-PyTorch
#          selective SSM fallback that runs on CPU).
model = HoloGNN(use_ssm=True)

# §8.2 multi-species proteomics — condition the proteomics head on a species id.
model = HoloGNN(num_species=5)          # then set data.species_id per sample

# Options compose; all default to the original V5 behaviour when omitted.
model = HoloGNN(fusion_mode="cross_attention", use_ssm=True, num_species=5)
```

| §8.2 item | Where | How to use |
|-----------|-------|-----------|
| (i) Cross-attention fusion | [`src/sequence_mixers.py`](src/sequence_mixers.py) `CrossAttentionFusion` | `HoloGNN(fusion_mode="cross_attention")` |
| (ii) Mamba state-space layers | [`src/sequence_mixers.py`](src/sequence_mixers.py) `SelectiveSSM` | `HoloGNN(use_ssm=True)` |
| (iii) Three-state head | [`src/heads.py`](src/heads.py) `ThreeStateStabilityHead` | `model((wt, mt), task="three_state")` |
| (iv) Multi-task MFI | [`src/heads.py`](src/heads.py) `MFIHead` | `model(data, task="mfi")` |
| Multi-species proteomics | [`src/heads.py`](src/heads.py) `ProteomicsHead(num_species>0)` | `HoloGNN(num_species=N)` |

> These are architectural scaffolds validated for correct forward/backward passes; training them to convergence on the full datasets (and the 2.8 M-PSM MassIVE-KB multi-species RT benchmark) is the remaining experimental work.

**Also in progress:** the interactive **web UI** — see [`../HOLOGNN_APP`](../HOLOGNN_APP) (real-time ΔΔG with confidence intervals, stability-landscape heatmaps, IDR ensemble plots, AlphaFold2 pLDDT/PAE comparison, and CSV/JSON/PDB export).

---

## Authors & citation

**Md Ahbab Hamid Khan, Md Wali Ullah Khan, Jamiul Hasan, Shadhin Nandi, Ahnaf Atique, Akib Bari, and Riasat Azim\***
Department of Computer Science & Engineering, United International University, Dhaka 1212, Bangladesh.
*\*Corresponding author.*

If you use Holo-GNN, please cite:

```bibtex
@article{khan2026holognn,
  title   = {Holo-GNN: a unified deep learning framework for protein
             stability, interaction, and expression prediction},
  author  = {Khan, Md Ahbab Hamid and Khan, Md Wali Ullah and Hasan, Jamiul and
             Nandi, Shadhin and Atique, Ahnaf and Bari, Akib and Azim, Riasat},
  year    = {2026},
  institution = {United International University}
}
```

---

## License

Released for academic and research use. See `LICENSE` (add your preferred license, e.g. MIT, before publishing).
