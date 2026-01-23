import torch

def build_graph_from_attention(attention_map, threshold=0.1):
    """
    Constructs a graph from the ESM-2 Attention Map.
    
    Args:
        attention_map (Tensor): Shape (Seq_Len, Seq_Len). 
                               Represents how much residue i 'looks at' residue j.
        threshold (float): Cutoff to define an 'Edge'.
    
    Returns:
        edge_index (LongTensor): Shape (2, Num_Edges). The connections.
    """
    seq_len = attention_map.shape[0]
    
    # 1. Symmetrize the map (If i looks at j, connect them both ways)
    # We add the transpose: A + A.T
    adj = attention_map + attention_map.t()
    
    # 2. Apply Threshold (Keep only strong connections)
    # We construct an edge if the attention score is high enough
    # This acts as a proxy for "3D Contact"
    rows, cols = torch.where(adj > threshold)
    
    # 3. Filter out Self-Loops (Don't connect a node to itself)
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    
    # 4. Create Edge Index
    edge_index = torch.stack([rows, cols], dim=0)
    
    return edge_index

def simple_linear_graph(seq_len):
    """
    Fallback: Connects residue i to i+1 (Linear Chain).
    Used if Attention is unavailable.
    """
    source = torch.arange(0, seq_len - 1)
    target = torch.arange(1, seq_len)
    
    # Bi-directional (i->i+1 AND i+1->i)
    rows = torch.cat([source, target])
    cols = torch.cat([target, source])
    
    return torch.stack([rows, cols], dim=0) 