# Migrating Holo-GNN to Vertex AI: Unified Jupyter Notebook Blueprint

This outline serves as a structural blueprint for migrating the Holo-GNN codebase into a single, cohesive Jupyter Notebook optimized for Google Cloud Vertex AI. The notebook is designed to read seamlessly as an executable research paper, moving logically from environment configuration to final figure generation.

---

## 1. Environment Setup & Unzipping Data

**Objective**: Configure the Vertex AI environment, install necessary dependencies (such as `torch`, `transformers`, `Bio`, `PyTorch Geometric`), and provision the dataset.

*   **System Check & Installations**: 
    *   Explanatory text detailing the required dependencies for geometric deep learning and language model tokenization (ESM-2).
    *   Placeholders for `!pip install` commands covering modules like `transformers` and Biopython.
*   **Hardware Verification**: 
    *   A block to verify CUDA parity and confirm the allocation of the NVIDIA T4 Tensor Core GPU using `torch.cuda.is_available()`.
*   **Data Provisioning**:
    *   Markdown framing the *Mega-scale cDNA* dataset as the primary benchmarking dataset for protein stability ($\Delta G$).
    *   Commands structured to unzip the dataset from `data.zip` (or Google Cloud Storage) into the local Vertex AI workspace `data/mega_scale_cdna/...`.

## 2. Model & Dataset Class Definitions

**Objective**: Consolidate the distributed Python architecture (`src/dataset.py` and `src/full_model.py`) into unified, readable notebook cells.

*   **The MegaScaleDataset Loader**:
    *   **Textual Context**: Introduce the `MegaScaleDataset` class. Explain its dual-purpose role: dynamically translating raw DNA sequences to protein representations (via Biopython) and tokenizing them using the pre-trained robust `facebook/esm2_t6_8M_UR50D` configuration.
    *   **Code Cell Placeholder**: Combine the imports and the class definition derived from `src/dataset.py`.
*   **Holo-GNN Unified Architecture**:
    *   **Textual Context**: Detail the architectural flow. Describe the sequence encoding through `HoloGNNBackbone`, pooling, and subsequent routing into the stability/IDR prediction space via `SiameseStabilityHead` or `ProteomicsHead`. 
    *   **Code Cell Placeholder**: A monolithic code cell redefining the components of `src/full_model.py` (and implicitly `backbone.py`/`heads.py` if needed) to ensure the notebook runs entirely independently of the `src/` directory.

## 3. T4 GPU Optimization (Batch Size 32, num_workers 8)

**Objective**: Calibrate hyperparameters specifically to exploit the memory and parallelization capabilities of the cloud-hosted T4 CPU/GPU combinations, removing legacy GTX 1050 Ti limitations.

*   **Scaling the Dataloaders**:
    *   **Textual Context**: Discuss the hardware upgrade narrative. Explain how transitioning to Vertex AI permits disabling gradient accumulation trickery (previously Batch Size 4 $\times$ 8 accumulation steps) in favor of a true, continuous `BATCH_SIZE = 32`.
    *   **Worker Allocation**: Detail the optimization of I/O latency by leveraging Vertex AI CPU cores. Justify the update from `num_workers=0` (local Windows constraint) to `num_workers=8` to ensure the T4 GPU is rapidly fed with complex tokenized sequences.
*   **Hyperparameter Initialization Block**:
    *   **Code Cell Placeholder**: Define global configuration constants including the learning rate ($1e-4$), MAX_SAMPLES, dataset splits, and initialization of the updated DataLoaders.

## 4. The Training Loop

**Objective**: Execute the primary model optimization mechanism utilizing the defined variables and architecture.

*   **Loss and Optimizer Configuration**:
    *   **Textual Context**: Briefly outline the selection of Mean Squared Error (`nn.MSELoss`) suited for the continuous $\Delta G$ regression target and the `Adam` optimizer.
*   **Epoch Execution Strategy**:
    *   **Textual Context**: Describe the feedforward procedure: mapping `input_ids`, parsing `attention_mask` parameters, setting the `idr` routing task, and tracking loss via `tqdm`. 
    *   **Refactored Logic**: Highlight the removal of the manual accumulator loop, yielding a cleaner, native PyTorch backward-pass workflow tailored for the new high-throughput batch settings.
*   **Code Cell Placeholder**: The `train` logic adapted from `train_final.py` (or `train_full_scale.py`), integrating standard `loss.backward()` and `optimizer.step()` unhindered by gradient chunking. Will also encompass the logic for saving the resulting `holognn_stability_final.pth` weights to the persistent Vertex AI disk.

## 5. Evaluation & Figure Generation

**Objective**: Systematically validate the learned capabilities of the model and generate publication-ready plots.

*   **Validation Inference**:
    *   **Textual Context**: Introduce the validation routine tracking predictive variance against the withheld 10% test-split.
    *   **Code Cell Placeholder**: A `torch.no_grad()` inference loop iterating through the `val_loader`, storing `y_pred` vs `y_actual`.
*   **Quantitative Metrics & Plotting**:
    *   **Textual Context**: Lay out the procedure to calculate foundational benchmarks like the Pearson correlation coefficient ($R$) and Mean Absolute Error (MAE) for stability evaluations. Mention diagnostic figures if applicable (ROC/Confusion Matrices from the distinct proteomics task pipeline).
    *   **Code Cell Placeholder**: The `matplotlib` / `seaborn` implementation plotting "Predicted vs Actual Stability". Equivalent code mapped directly from `generate_paper_figures.py` to ensure high-fidelity scatter plots output inline within the final Jupyter cells.
