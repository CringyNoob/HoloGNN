import torch.nn as nn
# We use explicit imports 'src.xxx' to avoid resolution errors
from src.backbone import HoloGNNBackbone
from src.heads import ProteomicsHead, SiameseStabilityHead, EnsembleIDRHead

class HoloGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize backbone (defaults are now 320)
        self.backbone = HoloGNNBackbone(hidden_dim=320, output_dim=320)
        
        # CHANGE THESE LINES: Explicitly set input_dim=320
        self.proteomics_head = ProteomicsHead(input_dim=320)
        self.siamese_head = SiameseStabilityHead(input_dim=320)
        self.idr_head = EnsembleIDRHead(input_dim=320)

    def forward(self, data, task="proteomics"):
        # 1. Extract Backbone Features
        # We pass the input_ids and mask. edge_index might be None.
        node_emb, graph_emb = self.backbone(data.input_ids, data.mask, data.edge_index)

        # 2. Route to Task
        if task == "proteomics":
            return self.proteomics_head(graph_emb)
            
        elif task == "idr":
            # For IDR/Stability, we predict the 'deltaG' or 'compaction'
            # Note: For the 'siamese' task, we would need two inputs.
            # For this simple training loop, we treat stability as a simple regression on one sequence.
            # (Later we will upgrade this to the full Siamese logic)
            return self.siamese_head.mlp(graph_emb) 
            
        return graph_emb