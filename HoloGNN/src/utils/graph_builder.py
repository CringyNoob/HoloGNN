import torch


# =============================================================================
# V6 — per-sample batched graph construction with edge weights
# =============================================================================
def build_batched_attention_graph(
    attentions: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int = 8,
    add_backbone: bool = True,
):
    """
    Build a SEPARATE contact graph for every sequence in the batch (V6).

    Unlike the V5 path---which averaged the whole batch into a single shared
    (L, L) topology---this constructs each protein's own graph from its own
    attention map, keeps the attention weight as an edge feature, uses a stable
    top-k neighbourhood instead of a brittle global threshold, and guarantees
    backbone (i, i+1) connectivity. Node indices are offset by ``b * L`` so the
    result indexes directly into the flattened ``(B*L, C)`` node tensor that the
    GAT layers operate on (standard PyG mini-batching).

    Args:
        attentions     : (B, L, L) per-sample attention map (already head-averaged).
        attention_mask : (B, L) with 1 = real token, 0 = padding.
        k              : top-k neighbours per node (clamped to the number of
                         available real neighbours).
        add_backbone   : also connect consecutive real residues (i, i+1).

    Returns:
        edge_index : LongTensor (2, E) global directed edges.
        edge_attr  : FloatTensor (E, 1) the symmetrised attention weight on each
                     edge (self-loops carry weight 1.0).
    """
    B, L, _ = attentions.shape
    device = attentions.device
    edges_src, edges_dst, weights = [], [], []

    for b in range(B):
        real = torch.nonzero(attention_mask[b] > 0, as_tuple=False).flatten()
        n = real.numel()
        if n == 0:
            continue
        offset = b * L

        # Self-loops for every real node (weight 1.0) so no real node is isolated.
        edges_src.append(real + offset)
        edges_dst.append(real + offset)
        weights.append(torch.ones(n, device=device))

        if n > 1:
            # Symmetrised attention restricted to this sample's real residues.
            adj = attentions[b][real][:, real]            # (n, n)
            adj = adj + adj.t()
            adj.fill_diagonal_(0.0)                        # no self in top-k

            kk = int(min(k, n - 1))
            if kk > 0:
                topw, topi = torch.topk(adj, kk, dim=1)    # (n, kk)
                src_local = torch.arange(n, device=device).unsqueeze(1).expand(-1, kk)
                edges_src.append(real[src_local.reshape(-1)] + offset)
                edges_dst.append(real[topi.reshape(-1)] + offset)
                weights.append(topw.reshape(-1))

            if add_backbone:
                # Consecutive real positions, both directions.
                a = real[:-1]
                c = real[1:]
                bw = adj[torch.arange(n - 1, device=device),
                         torch.arange(1, n, device=device)]
                edges_src.append(torch.cat([a, c]) + offset)
                edges_dst.append(torch.cat([c, a]) + offset)
                weights.append(torch.cat([bw, bw]))

    if not edges_src:                                      # degenerate batch
        ei = torch.empty(2, 0, dtype=torch.long, device=device)
        ea = torch.empty(0, 1, device=device)
        return ei, ea

    edge_index = torch.stack([torch.cat(edges_src), torch.cat(edges_dst)], dim=0).long()
    edge_attr = torch.cat(weights).unsqueeze(-1).float()
    return edge_index, edge_attr


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