from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.35):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(adj, x)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        return self.lin2(h).squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.35):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels * 2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_mean, x)
        h = F.relu(self.lin1(torch.cat([x, neigh], dim=1)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        neigh_h = torch.sparse.mm(adj_mean, h)
        return self.lin2(torch.cat([h, neigh_h], dim=1)).squeeze(-1)


class SimpleGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.35):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.att_src1 = nn.Parameter(torch.empty(hidden_channels))
        self.att_dst1 = nn.Parameter(torch.empty(hidden_channels))
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_uniform_(self.att_src1.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst1.unsqueeze(0))

    def _attend(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        e = F.leaky_relu((h[src] * self.att_src1).sum(dim=1) + (h[dst] * self.att_dst1).sum(dim=1), negative_slope=0.2)
        max_per_dst = torch.full((h.size(0),), -torch.inf, dtype=h.dtype, device=h.device)
        max_per_dst.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[dst]).clamp_max(1e6)
        denom = torch.zeros((h.size(0),), dtype=h.dtype, device=h.device)
        denom.scatter_add_(0, dst, exp_e)
        alpha = exp_e / denom[dst].clamp_min(1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * alpha.unsqueeze(1))
        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin1(h)
        h = F.elu(self._attend(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h).squeeze(-1)

