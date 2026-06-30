"""
src/full_model.py
=================
HoloGNN Full Model — Multi-task Protein Structure & Function Predictor.

Supported tasks:
  'proteomics'  — Single-sequence retention-time regression (MassIVE-KB).
  'stability'   — Single-sequence absolute ΔG regression (MegaScale).
  'mfi'         — Auxiliary mean-fluorescence-intensity regression.
  'idr'         — Siamese ΔΔG stability regression (FireProtDB/MegaScale).
  'three_state' — Three-class stability classification (destab/neutral/stab).
  default       — Returns raw graph embedding for downstream fine-tuning.
"""

import torch
import torch.nn as nn
from src.backbone import HoloGNNBackbone
from src.heads import (
    ProteomicsHead,
    SiameseStabilityHead,
    StabilityScoreHead,
    ThreeStateStabilityHead,
    MFIHead,
    StabilityRegressionHead,
)


class HoloGNN(nn.Module):
    """
    HoloGNN — Multi-task Protein Structure & Function Predictor.

    Supported tasks
    ---------------
    'proteomics'
        Single-sequence retention-time regression (MassIVE-KB pre-training).
        Input : data        — single DataBatch
        Output: scalar prediction per sample

    'idr'
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
                 mech_feature_dim: int = 6, top_k: int = 8,
                 antisym_head: bool = False, heteroscedastic: bool = False,
                 freeze_esm: bool = False, freeze_esm_layers: int = 0):
        """
        Args:
            fusion_mode       : "concat" (default) or "cross_attention".
            use_ssm           : enable the Selective-SSM / Mamba mixer.
            num_species       : >0 enables multi-species proteomics conditioning.
            pool              : "attention" (default) or "mean" graph pooling.
            graph_mode        : "per_sample" (default) or "shared" graph.
            mech_feature_dim  : number of mechanistic channels (default 6).
            top_k             : neighbours per node in per-sample graphs.
            antisym_head      : use the antisymmetric-by-construction ΔΔG head.
            heteroscedastic   : ΔΔG head also predicts σ (calibrated CIs).
            freeze_esm        : freeze the whole ESM-2 encoder.
            freeze_esm_layers : freeze ESM-2 embeddings + first N encoder layers.
        """
        super().__init__()
        self.use_antisym_head = antisym_head
        self.heteroscedastic  = heteroscedastic

        # Backbone
        self.backbone = HoloGNNBackbone(
            output_dim=320, fusion_mode=fusion_mode, use_ssm=use_ssm,
            pool=pool, graph_mode=graph_mode, mech_feature_dim=mech_feature_dim,
            top_k=top_k, freeze_esm=freeze_esm, freeze_esm_layers=freeze_esm_layers,
        )

        # Task heads — all receive a 320-dim graph embedding.
        self.proteomics_head  = ProteomicsHead(input_dim=320, num_species=num_species)
        self.siamese_head     = SiameseStabilityHead(input_dim=320, heteroscedastic=heteroscedastic)
        self.three_state_head = ThreeStateStabilityHead(input_dim=320)
        self.mfi_head         = MFIHead(input_dim=320)
        self.stability_head   = StabilityRegressionHead(input_dim=320)

        # Antisymmetric-by-construction ΔΔG head (opt-in).
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
                     .mechanistic_features  — (B, L, mech_feature_dim)
                     .edge_index            — (2, E) or None
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
            'three_state' → Tensor (B, 3)   class logits (destab/neutral/stab)
            default       → Tensor (B, 320) raw embedding
        """

        # Task: proteomics  (single-sequence regression; optional species id)
        if task == "proteomics":
            z = self._encode(data)
            return self.proteomics_head(z, getattr(data, "species_id", None))

        # Task: stability  (single-sequence absolute ΔG regression — MegaScale)
        if task == "stability":
            z = self._encode(data)
            return self.stability_head(z)

        # Task: mfi  (auxiliary mean-fluorescence-intensity regression)
        if task == "mfi":
            z = self._encode(data)
            return self.mfi_head(z)

        # Task: three_state  (destabilising / neutral / stabilising)
        if task == "three_state":
            data_wt, data_mt = data
            z_wt = self._encode(data_wt)
            z_mt = self._encode(data_mt)
            return self.three_state_head(z_wt, z_mt)

        # Task: idr  (Siamese ΔΔG pass)
        elif task == "idr":
            data_wt, data_mt = data

            z_wt = self._encode(data_wt)
            z_mt = self._encode(data_mt)

            # Antisymmetric-by-construction head
            if self.use_antisym_head:
                ddg_fwd, logvar = self.score_head(z_wt, z_mt)
                ddg_rev, _      = self.score_head(z_mt, z_wt)
                if self.heteroscedastic:
                    return ddg_fwd, ddg_rev, logvar
                return ddg_fwd, ddg_rev

            # Standard Siamese head
            if self.heteroscedastic:
                mu_fwd, logvar = self.siamese_head(z_wt, z_mt)
                mu_rev, _      = self.siamese_head(z_mt, z_wt)
                return mu_fwd, mu_rev, logvar

            dG_wt_to_mt = self.siamese_head(z_wt, z_mt)
            dG_mt_to_wt = self.siamese_head(z_mt, z_wt)
            return dG_wt_to_mt, dG_mt_to_wt

        # Default: return raw graph embedding
        return self._encode(data)