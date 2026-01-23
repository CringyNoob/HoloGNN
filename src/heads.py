import torch
import torch.nn as nn

# 1. Proteomics Head (MassIVE-KB) 
# Predicts Retention Time (Scalar)
class ProteomicsHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output: Retention Time
        )
    
    def forward(self, graph_embedding):
        return self.regressor(graph_embedding)

# 2. Siamese Stability Head (FireProtDB) [cite: 198]
# Predicts DeltaDeltaG using antisymmetry
class SiameseStabilityHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output: DeltaDeltaG
        )

    def forward(self, embedding_wildtype, embedding_mutant):
        # Calculate difference vector [cite: 202]
        diff = embedding_mutant - embedding_wildtype
        return self.mlp(diff)

# 3. Ensemble Dimension Head (ClinVar/IDR) [cite: 207]
# Predicts Mean and Sigma for Radius of Gyration
class EnsembleIDRHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        # Branch for Mean (Mu)
        self.mu_layer = nn.Linear(input_dim, 1)
        # Branch for Standard Deviation (Sigma)
        self.sigma_layer = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus() # Ensures sigma is positive

    def forward(self, graph_embedding):
        mu = self.mu_layer(graph_embedding)
        sigma = self.softplus(self.sigma_layer(graph_embedding))
        return mu, sigma