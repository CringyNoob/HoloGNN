"""
src/heads.py
============
HoloGNN task heads.

Original heads (V4/V5)
    • ProteomicsHead         — retention-time / expression regression
    • SiameseStabilityHead   — ΔΔG regression from the (z_mt − z_wt) difference
    • EnsembleIDRHead        — radius-of-gyration ensemble (μ, σ)

Future-work heads (paper §8.2, now implemented)
    • ThreeStateStabilityHead — stabilising / neutral / destabilising (3-class)
    • MFIHead                  — mean-fluorescence-intensity auxiliary regression
    • StabilityRegressionHead  — single-sequence absolute ΔG regression

All heads receive a 320-dim graph embedding from the backbone.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Proteomics Head (MassIVE-KB) — Retention Time / expression (scalar)
#    §8.2 multi-species: optionally conditions on a per-sample species id via a
#    learned species embedding (backward compatible — disabled when num_species=0).
# ---------------------------------------------------------------------------
class ProteomicsHead(nn.Module):
    def __init__(self, input_dim: int = 512, num_species: int = 0,
                 species_dim: int = 16):
        super().__init__()
        self.num_species = num_species
        in_dim = input_dim
        if num_species and num_species > 0:
            self.species_embedding = nn.Embedding(num_species, species_dim)
            in_dim = input_dim + species_dim
        else:
            self.species_embedding = None

        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output: Retention Time
        )

    def forward(self, graph_embedding, species_id: torch.Tensor | None = None):
        x = graph_embedding
        if self.species_embedding is not None and species_id is not None:
            spec = self.species_embedding(species_id)
            if spec.dim() == graph_embedding.dim():
                x = torch.cat([graph_embedding, spec], dim=-1)
            else:  # (B,) species_id → (B, species_dim)
                x = torch.cat([graph_embedding, spec.view(graph_embedding.size(0), -1)], dim=-1)
        return self.regressor(x)


# ---------------------------------------------------------------------------
# 2. Siamese Stability Head (FireProtDB / MegaScale) — ΔΔG via antisymmetry
# ---------------------------------------------------------------------------
class SiameseStabilityHead(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output: ΔΔG
        )

    def forward(self, embedding_wildtype, embedding_mutant):
        diff = embedding_mutant - embedding_wildtype   # difference vector
        return self.mlp(diff)


# ---------------------------------------------------------------------------
# 3. Ensemble Dimension Head (ClinVar / IDR) — μ, σ of Radius of Gyration
# ---------------------------------------------------------------------------
class EnsembleIDRHead(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.mu_layer    = nn.Linear(input_dim, 1)
        self.sigma_layer = nn.Linear(input_dim, 1)
        self.softplus    = nn.Softplus()  # keeps σ positive

    def forward(self, graph_embedding):
        mu    = self.mu_layer(graph_embedding)
        sigma = self.softplus(self.sigma_layer(graph_embedding))
        return mu, sigma


# ===========================================================================
# Future-work heads (paper §8.2)
# ===========================================================================

# ---------------------------------------------------------------------------
# 4. Three-State Classification Head (§8.2-iii)
#    Explicitly classifies a mutation as destabilising / neutral / stabilising,
#    handling experimental noise near ΔΔG ≈ 0.  Operates on the Siamese
#    difference embedding so it shares the antisymmetric inductive bias.
#
#    Class index convention:  0 = destabilising, 1 = neutral, 2 = stabilising.
# ---------------------------------------------------------------------------
class ThreeStateStabilityHead(nn.Module):
    NEUTRAL_BAND_KCAL = 0.5  # |ΔΔG| < band ⇒ neutral; used to derive labels

    def __init__(self, input_dim: int = 512, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),  # logits
        )

    def forward(self, embedding_wildtype, embedding_mutant):
        diff = embedding_mutant - embedding_wildtype
        return self.classifier(diff)  # (B, num_classes) logits

    @classmethod
    def labels_from_ddg(cls, ddg: torch.Tensor) -> torch.Tensor:
        """Map continuous ΔΔG (B,) → class indices using the neutral band.
        Convention: ΔΔG > +band → stabilising(2); ΔΔG < −band → destabilising(0)."""
        band = cls.NEUTRAL_BAND_KCAL
        labels = torch.ones_like(ddg, dtype=torch.long)        # neutral = 1
        labels[ddg >  band] = 2                                 # stabilising
        labels[ddg < -band] = 0                                 # destabilising
        return labels


# ---------------------------------------------------------------------------
# 5. MFI Multi-Task Head (§8.2-iv)
#    Mean-fluorescence-intensity regression from suppressor libraries, used as
#    an auxiliary training signal alongside the stability objective.
# ---------------------------------------------------------------------------
class MFIHead(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output: mean fluorescence intensity
        )

    def forward(self, graph_embedding):
        return self.regressor(graph_embedding)


# ---------------------------------------------------------------------------
# 6. Single-Sequence Stability Regression Head
#    Absolute ΔG regression for the MegaScale 'stability' task (one sequence,
#    one ΔG label).  Lets the model train on single-sequence stability without
#    the Siamese pair machinery.
# ---------------------------------------------------------------------------
class StabilityRegressionHead(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output: ΔG (kcal/mol)
        )

    def forward(self, graph_embedding):
        return self.regressor(graph_embedding)
