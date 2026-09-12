from __future__ import annotations

import torch


def prepare_edges(edge_index: torch.Tensor, num_nodes: int, bidirectional: bool = True, self_loops: bool = True) -> torch.Tensor:
    edges = edge_index.long()
    if bidirectional:
        rev = torch.stack([edges[1], edges[0]], dim=0)
        edges = torch.cat([edges, rev], dim=1)
    if self_loops:
        loop = torch.arange(num_nodes, dtype=torch.long, device=edges.device)
        loops = torch.stack([loop, loop], dim=0)
        edges = torch.cat([edges, loops], dim=1)
    return torch.unique(edges, dim=1)


def normalized_sparse_adj(edge_index: torch.Tensor, num_nodes: int, mode: str = "gcn") -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    deg_dst = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
    if mode == "mean":
        weight = 1.0 / deg_dst[dst]
    elif mode == "gcn":
        deg_src = torch.bincount(src, minlength=num_nodes).float().clamp_min(1.0)
        weight = deg_src[src].pow(-0.5) * deg_dst[dst].pow(-0.5)
    else:
        raise ValueError(f"Unknown adjacency normalization mode: {mode}")
    indices = torch.stack([dst, src], dim=0)
    return torch.sparse_coo_tensor(indices, weight, (num_nodes, num_nodes)).coalesce()

