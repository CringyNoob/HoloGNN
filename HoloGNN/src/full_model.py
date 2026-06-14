"""
src/full_model.py
=================
HoloGNN Full Model — Evolution History
----------------------------------------
V1.0  Single-sequence forward pass; 'idr' task called siamese_head.mlp directly.
V4.0  True Siamese Pass: 'idr' task accepts a (data_wt, data_mt) tuple, passes
      each through the backbone independently, and feeds (z_wt, z_mt) into
      SiameseStabilityHead.  This unlocks the AntisymmetricLoss in loss.py.
"""

import torch
import torch.nn as nn
from src.backbone import HoloGNNBackbone
from src.heads import (
    ProteomicsHead,
    SiameseStabilityHead,
    StabilityScoreHead,
    EnsembleIDRHead,
    ThreeStateStabilityHead,
    MFIHead,
    StabilityRegressionHead,
)


class HoloGNN(nn.Module):
    """
    HoloGNN — Multi-task Protein Structure & Function Predictor
    Version 4.0

    Supported tasks
    ---------------
    'proteomics'
        Single-sequence retention-time regression (MassIVE-KB pre-training).
        Input : data        — single DataBatch
        Output: scalar prediction per sample

    'idr'  [V4.0 upgrade]
        Siamese ΔΔG stability regression (MegaScale / FireProtDB).
        Input : data        — tuple (data_wt, data_mt)
                              Each is a DataBatch with .input_ids, .mask,
                              .edge_index, .mechanistic_features
        Output: (dG_wt_to_mt, dG_mt_to_wt)
                Both are scalar predictions used by AntisymmetricLoss.

    default
        Returns the raw graph embedding for downstream fine-tuning.
    """

    def __init__(self, fusion_mode: str = "concat", use_ssm: bool = False,
                 num_species: int = 0,
                 pool: str = "attention", graph_mode: str = "per_sample",
                 mech_feature_dim: int = 3, top_k: int = 8,
                 antisym_head: bool = False, heteroscedastic: bool = False,
                 freeze_esm: bool = False, freeze_esm_layers: int = 0):
        """
        Args:
            fusion_mode       : "concat" (default) or "cross_attention" (§8.2-i).
            use_ssm           : enable the Selective-SSM / Mamba mixer (§8.2-ii).
            num_species       : >0 enables multi-species proteomics conditioning.
            pool              : [V6] "attention" (default) or "mean" graph pooling.
            graph_mode        : [V6] "per_sample" (default) or "shared" (V5) graph.
            mech_feature_dim  : 3 (V5) or 6 (expanded protein-only features, V6).
            top_k             : [V6] neighbours per node in per-sample graphs.
            antisym_head      : [V6] use the antisymmetric-by-construction ΔΔG head.
            heteroscedastic   : [V6] ΔΔG head also predicts σ (calibrated CIs).
            freeze_esm        : freeze the whole ESM-2 encoder.
            freeze_esm_layers : freeze ESM-2 embeddings + first N encoder layers.
        The V6 defaults (attention pooling, per-sample graphs) improve batched
        training; everything else defaults to the original V5 behaviour.
        """
        super().__init__()
        self.use_antisym_head = antisym_head
        self.heteroscedastic  = heteroscedastic

        # ------------------------------------------------------------------
        # Backbone  (V6: mask-aware pooling, per-sample attention graphs)
        # ------------------------------------------------------------------
        self.backbone = HoloGNNBackbone(
            output_dim=320, fusion_mode=fusion_mode, use_ssm=use_ssm,
            pool=pool, graph_mode=graph_mode, mech_feature_dim=mech_feature_dim,
            top_k=top_k, freeze_esm=freeze_esm, freeze_esm_layers=freeze_esm_layers,
        )

        # ------------------------------------------------------------------
        # Task heads — all receive a 320-dim graph embedding.
        # ------------------------------------------------------------------
        self.proteomics_head = ProteomicsHead(input_dim=320, num_species=num_species)
        self.siamese_head    = SiameseStabilityHead(input_dim=320, heteroscedastic=heteroscedastic)
        self.idr_head        = EnsembleIDRHead(input_dim=320)

        # §8.2 future-work heads
        self.three_state_head = ThreeStateStabilityHead(input_dim=320)
        self.mfi_head         = MFIHead(input_dim=320)
        self.stability_head   = StabilityRegressionHead(input_dim=320)

        # [V6] Antisymmetric-by-construction ΔΔG head (opt-in).
        self.score_head = (StabilityScoreHead(input_dim=320, heteroscedastic=heteroscedastic)
                           if antisym_head else None)

    # -----------------------------------------------------------------------
    def _encode(self, data) -> torch.Tensor:
        """
        Shared helper: run a single DataBatch through the backbone and return
        the graph-level embedding z of shape (B, 320).

        Args:
            data : object with attributes
                     .input_ids             — (B, L)
                     .mask                  — (B, L)
                     .mechanistic_features  — (B, L, 3)  [V3.0]
                     .edge_index            — (2, E) or None [V2.0]
        """
        _, graph_emb = self.backbone(
            input_ids            = data.input_ids,
            attention_mask       = data.mask,
            mechanistic_features = data.mechanistic_features,
            edge_index           = data.edge_index,
        )
        return graph_emb   # (B, 320)

    # -----------------------------------------------------------------------
    def forward(self, data, task: str = "proteomics"):
        """
        Args:
            data : DataBatch  —or—  tuple(DataBatch, DataBatch) for the paired
                   tasks ('idr', 'three_state').
            task : 'proteomics' | 'stability' | 'mfi' | 'idr' | 'three_state'.

        Returns:
            'proteomics'  → Tensor (B, 1)   retention-time / expression
            'stability'   → Tensor (B, 1)   single-sequence absolute ΔG
            'mfi'         → Tensor (B, 1)   mean fluorescence intensity
            'idr'         → tuple( dG_wt_to_mt (B,1), dG_mt_to_wt (B,1) )
            'pathogenicity'→ Tensor (B,)    pathogenicity logit (ClinVar)
            'three_state' → Tensor (B, 3)   class logits (destab/neutral/stab)
            default       → Tensor (B, 320) raw embedding
        """

        # ------------------------------------------------------------------
        # Task: proteomics  (single-sequence regression; optional species id)
        # ------------------------------------------------------------------
        if task == "proteomics":
            z = self._encode(data)
            return self.proteomics_head(z, getattr(data, "species_id", None))

        # ------------------------------------------------------------------
        # Task: stability  (single-sequence absolute ΔG regression — MegaScale)
        # ------------------------------------------------------------------
        if task == "stability":
            z = self._encode(data)
            return self.stability_head(z)

        # ------------------------------------------------------------------
        # Task: mfi  (§8.2-iv — auxiliary mean-fluorescence-intensity regression)
        # ------------------------------------------------------------------
        if task == "mfi":
            z = self._encode(data)
            return self.mfi_head(z)

        # ------------------------------------------------------------------
        # Task: three_state  (§8.2-iii — destabilising / neutral / stabilising)
        #   data is a (data_wt, data_mt) tuple, like 'idr'.
        # ------------------------------------------------------------------
        if task == "three_state":
            data_wt, data_mt = data
            z_wt = self._encode(data_wt)
            z_mt = self._encode(data_mt)
            return self.three_state_head(z_wt, z_mt)   # (B, 3) logits

        # ------------------------------------------------------------------
        # Task: pathogenicity  — ClinVar benign/pathogenic CLASSIFICATION.
        #   data is a (data_wt, data_mt) tuple (REF / ALT allele contexts).
        #   Uses the dedicated EnsembleIDRHead on the Siamese difference embedding
        #   to emit a single pathogenicity logit, so it can be trained directly on
        #   ClinVar labels (BCEWithLogitsLoss) instead of reusing the ΔΔG head as a
        #   sigmoid(-ΔΔG) proxy.  Returns raw logits (B,) — apply sigmoid for prob.
        # ------------------------------------------------------------------
        if task == "pathogenicity":
            data_wt, data_mt = data
            z_wt = self._encode(data_wt)
            z_mt = self._encode(data_mt)
            logit, _sigma = self.idr_head(z_mt - z_wt)   # (B, 1)
            return logit.squeeze(-1)                       # (B,)

        # ------------------------------------------------------------------
        # Task: idr  [V4.0 — True Siamese Pass]
        #
        # data must be a (data_wt, data_mt) tuple.
        # Each is passed through the shared backbone independently.
        # The resulting embeddings (z_wt, z_mt) are fed into SiameseStabilityHead
        # in BOTH directions.  This exposes the antisymmetry constraint that
        # AntisymmetricLoss enforces during training.
        #
        # Biological motivation:
        #   ΔΔG(WT→MT) should equal −ΔΔG(MT→WT).  The Siamese head computes
        #   the difference vector (z_mt − z_wt), so calling it in both orders
        #   gives predictions that AntisymmetricLoss can penalise for asymmetry.
        # ------------------------------------------------------------------
        elif task == "idr":
            data_wt, data_mt = data   # unpack tuple

            # Independent backbone passes — each sees its own attention graph
            z_wt = self._encode(data_wt)   # (B, 320)
            z_mt = self._encode(data_mt)   # (B, 320)

            # [V6] Antisymmetric-by-construction head: ΔΔG = s(z_mt) − s(z_wt),
            # so the reverse is exactly the negation (antisymmetry holds by design).
            if self.use_antisym_head:
                ddg_fwd, logvar = self.score_head(z_wt, z_mt)
                ddg_rev, _      = self.score_head(z_mt, z_wt)   # == −ddg_fwd
                if self.heteroscedastic:
                    return ddg_fwd, ddg_rev, logvar            # (B,1),(B,1),(B,1)
                return ddg_fwd, ddg_rev

            # Standard Siamese head (V4/V5).  Heteroscedastic adds a variance head.
            if self.heteroscedastic:
                mu_fwd, logvar = self.siamese_head(z_wt, z_mt)
                mu_rev, _      = self.siamese_head(z_mt, z_wt)
                return mu_fwd, mu_rev, logvar

            dG_wt_to_mt = self.siamese_head(z_wt, z_mt)   # (B, 1)
            dG_mt_to_wt = self.siamese_head(z_mt, z_wt)   # (B, 1)
            return dG_wt_to_mt, dG_mt_to_wt

        # ------------------------------------------------------------------
        # Default: return raw graph embedding
        # ------------------------------------------------------------------
        return self._encode(data)