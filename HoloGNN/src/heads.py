"""
src/heads.py
============
HoloGNN task heads.

All heads receive a 320-dim graph embedding from the backbone.

    • ProteomicsHead          — retention-time / expression regression
    • SiameseStabilityHead    — ΔΔG regression from the (z_mt − z_wt) difference
    • StabilityScoreHead      — antisymmetric-by-construction ΔΔG head
    • ThreeStateStabilityHead — stabilising / neutral / destabilising (3-class)
    • MFIHead                 — mean-fluorescence-intensity auxiliary regression
    • StabilityRegressionHead — single-sequence absolute ΔG regression
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Proteomics Head (MassIVE-KB) — Retention Time / expression (scalar)
# ---------------------------------------------------------------------------
class ProteomicsHead(nn.Module):
    def __init__(self, input_dim: int = 320, num_species: int = 0,
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
            nn.Linear(256, 1),
        )

    def forward(self, graph_embedding, species_id: torch.Tensor | None = None):
        x = graph_embedding
        if self.species_embedding is not None and species_id is not None:
            spec = self.species_embedding(species_id)
            if spec.dim() == graph_embedding.dim():
                x = torch.cat([graph_embedding, spec], dim=-1)
            else:
                x = torch.cat([graph_embedding, spec.view(graph_embedding.size(0), -1)], dim=-1)
        return self.regressor(x)


# ---------------------------------------------------------------------------
# 2. Siamese Stability Head (FireProtDB / MegaScale) — ΔΔG via antisymmetry
#    Optional heteroscedastic output (μ, log σ²) for calibrated uncertainty.
# ---------------------------------------------------------------------------
class SiameseStabilityHead(nn.Module):
    def __init__(self, input_dim: int = 320, heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        out_dim = 2 if heteroscedastic else 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, embedding_wildtype, embedding_mutant):
        diff = embedding_mutant - embedding_wildtype
        out = self.mlp(diff)
        if self.heteroscedastic:
            return out[..., :1], out[..., 1:2]
        return out


# ---------------------------------------------------------------------------
# 3. Antisymmetric-by-construction stability head
#    ΔΔG(wt→mt) = s(z_mt) − s(z_wt), exactly antisymmetric by design.
# ---------------------------------------------------------------------------
class StabilityScoreHead(nn.Module):
    def __init__(self, input_dim: int = 320, heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        out_dim = 2 if heteroscedastic else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def score(self, z):
        o = self.net(z)
        if self.heteroscedastic:
            return o[..., :1], o[..., 1:2]
        return o, None

    def forward(self, embedding_wildtype, embedding_mutant):
        mu_wt, lv_wt = self.score(embedding_wildtype)
        mu_mt, lv_mt = self.score(embedding_mutant)
        ddg = mu_mt - mu_wt
        if self.heteroscedastic:
            var    = torch.exp(lv_wt) + torch.exp(lv_mt)
            logvar = torch.log(var + 1e-8)
            return ddg, logvar
        return ddg, None


# ---------------------------------------------------------------------------
# 4. Three-State Classification Head
# ---------------------------------------------------------------------------
class ThreeStateStabilityHead(nn.Module):
    NEUTRAL_BAND_KCAL = 0.5

    def __init__(self, input_dim: int = 320, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, embedding_wildtype, embedding_mutant):
        diff = embedding_mutant - embedding_wildtype
        return self.classifier(diff)

    @classmethod
    def labels_from_ddg(cls, ddg: torch.Tensor) -> torch.Tensor:
        band = cls.NEUTRAL_BAND_KCAL
        labels = torch.ones_like(ddg, dtype=torch.long)
        labels[ddg >  band] = 2
        labels[ddg < -band] = 0
        return labels


# ---------------------------------------------------------------------------
# 5. MFI Multi-Task Head — mean fluorescence intensity regression
# ---------------------------------------------------------------------------
class MFIHead(nn.Module):
    def __init__(self, input_dim: int = 320):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, graph_embedding):
        return self.regressor(graph_embedding)


# ---------------------------------------------------------------------------
# 6. Single-Sequence Stability Regression Head — absolute ΔG for MegaScale
# ---------------------------------------------------------------------------
class StabilityRegressionHead(nn.Module):
    def __init__(self, input_dim: int = 320):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, graph_embedding):
        return self.regressor(graph_embedding)
