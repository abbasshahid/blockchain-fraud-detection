"""Re-executable wrapper around a trained GNN checkpoint.

Everything in the explanation layer that needs a counterfactual -- "what would
the model have predicted if this feature were absent, or if this payment link
did not exist?" -- goes through :class:`GnnProbe`.  It reloads a saved
checkpoint, rebuilds the exact architecture and adjacency used at training
time, and exposes forward passes under feature and edge interventions.

Interventions are deliberately local: only the target node's feature vector or
only its incident edges are modified, so the resulting probability change is
attributable to that node's own evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from blockchain_fraud.models.gnn import GCN, GraphSAGE, SimpleGAT
from blockchain_fraud.models.graph_utils import normalized_sparse_adj, prepare_edges
from blockchain_fraud.utils.io import load_graph


def _build_model(name: str, in_channels: int, hidden: int, dropout: float) -> torch.nn.Module:
    if name == "gcn":
        return GCN(in_channels, hidden, dropout)
    if name == "graphsage":
        return GraphSAGE(in_channels, hidden, dropout)
    if name == "gat":
        return SimpleGAT(in_channels, hidden, dropout)
    raise ValueError(f"Unsupported GNN model: {name}")


class GnnProbe:
    """Loads a trained GNN and answers counterfactual forward-pass queries."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        processed_path: str | Path,
    ) -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config = payload["config"]
        train_cfg = config.get("training", {})
        self.model_name = model_name
        self.graph = load_graph(processed_path)
        self.x = self.graph["x"].clone()
        self.num_nodes = self.x.size(0)
        self.base_edge_index = prepare_edges(
            self.graph["edge_index"],
            self.num_nodes,
            bidirectional=bool(train_cfg.get("bidirectional_edges", True)),
            self_loops=True,
        )
        self.model = _build_model(
            model_name,
            in_channels=self.x.size(1),
            hidden=int(train_cfg.get("hidden_channels", 32)),
            dropout=float(train_cfg.get("dropout", 0.35)),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self._adj_cache: dict[int, torch.Tensor | None] = {}
        self.feature_names: list[str] = list(payload.get("feature_names", self.graph["feature_names"]))

    # -- adjacency -------------------------------------------------------
    def _adj_for(self, edge_index: torch.Tensor) -> torch.Tensor | None:
        if self.model_name == "gcn":
            return normalized_sparse_adj(edge_index, self.num_nodes, "gcn")
        if self.model_name == "graphsage":
            return normalized_sparse_adj(edge_index, self.num_nodes, "mean")
        return None

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if self.model_name in {"gcn", "graphsage"}:
            return self.model(x, adj)
        return self.model(x, edge_index)

    # -- queries ---------------------------------------------------------
    def base_probabilities(self) -> np.ndarray:
        from scipy.special import expit

        return expit(self.base_logits())

    def base_logits(self) -> np.ndarray:
        adj = self._adj_for(self.base_edge_index)
        with torch.no_grad():
            logits = self._forward(self.x, self.base_edge_index, adj)
        return logits.numpy()

    def probability_with_masked_features(self, node: int, feature_idx: Sequence[int], baseline: float = 0.0) -> float:
        """Probability for ``node`` after neutralising selected features of that node only.

        Features are standardised with training-period statistics, so the
        training-period mean is exactly 0; setting a feature to 0 is therefore
        the natural "value unknown" baseline.
        """
        x = self.x.clone()
        if len(feature_idx):
            x[node, list(feature_idx)] = baseline
        adj = self._adj_for(self.base_edge_index)
        with torch.no_grad():
            logits = self._forward(x, self.base_edge_index, adj)
        return float(torch.sigmoid(logits[node]).item())

    def probability_with_only_features(self, node: int, feature_idx: Sequence[int], baseline: float = 0.0) -> float:
        """Probability when *only* the selected features of ``node`` are retained."""
        x = self.x.clone()
        keep = torch.zeros(x.size(1), dtype=torch.bool)
        if len(feature_idx):
            keep[list(feature_idx)] = True
        row = torch.full((x.size(1),), baseline, dtype=x.dtype)
        row[keep] = self.x[node][keep]
        x[node] = row
        adj = self._adj_for(self.base_edge_index)
        with torch.no_grad():
            logits = self._forward(x, self.base_edge_index, adj)
        return float(torch.sigmoid(logits[node]).item())

    def probability_with_removed_edges(self, node: int, drop: Iterable[tuple[int, int]] | None = None) -> float:
        """Probability after removing incident edges (all of them when ``drop`` is None).

        The node's self-loop is preserved so that the model still sees the
        transaction itself; only the payment links to other transactions go away.
        """
        edges = self.base_edge_index
        if drop is None:
            incident = ((edges[0] == node) | (edges[1] == node)) & (edges[0] != edges[1])
            keep = ~incident
        else:
            drop_set = {(int(a), int(b)) for a, b in drop}
            src = edges[0].tolist()
            dst = edges[1].tolist()
            keep = torch.tensor([(s, d) not in drop_set for s, d in zip(src, dst)], dtype=torch.bool)
        edge_index = edges[:, keep]
        adj = self._adj_for(edge_index)
        with torch.no_grad():
            logits = self._forward(self.x, edge_index, adj)
        return float(torch.sigmoid(logits[node]).item())

    # -- batched queries -------------------------------------------------
    def independent_groups(self, nodes: Sequence[int], hops: int = 2) -> list[list[int]]:
        """Partition nodes so that no two nodes in a group are within ``hops`` hops.

        A feature intervention on node ``u`` can only change the readout of node
        ``v`` when ``v`` lies inside ``u``'s receptive field, which for the
        two-layer models used here is two hops.  Nodes that are mutually further
        apart can therefore be perturbed in a single forward pass without
        interfering, which is what makes the deletion curves affordable.

        The conflict test is done with sparse matrix products rather than by
        expanding each node's frontier: hub transactions have very large two-hop
        neighbourhoods, and materialising them is what dominates the cost.
        """
        import scipy.sparse as sp

        nodes = [int(n) for n in nodes]
        edges = self.base_edge_index.numpy()
        n = self.num_nodes
        adj = sp.coo_matrix(
            (np.ones(edges.shape[1] * 2, dtype=bool),
             (np.concatenate([edges[0], edges[1]]), np.concatenate([edges[1], edges[0]]))),
            shape=(n, n),
        ).tocsr()

        # Rows of the selected nodes, expanded `hops` times, then restricted
        # back to the selected columns: a 100x100 conflict matrix.
        selector = sp.csr_matrix(
            (np.ones(len(nodes), dtype=bool), (np.arange(len(nodes)), nodes)), shape=(len(nodes), n)
        )
        reach = selector.copy()
        for _ in range(hops):
            reach = ((reach + reach @ adj) > 0).astype(bool)
        conflict = (reach[:, nodes].toarray() > 0)
        conflict = conflict | conflict.T
        np.fill_diagonal(conflict, False)

        groups: list[list[int]] = []
        assigned: list[list[int]] = []
        for i, node in enumerate(nodes):
            for slot, members in enumerate(assigned):
                if not conflict[i, members].any():
                    members.append(i)
                    groups[slot].append(node)
                    break
            else:
                assigned.append([i])
                groups.append([node])
        return groups

    def batch_masked_probabilities(
        self,
        masks: dict[int, Sequence[int]],
        groups: list[list[int]],
        keep_only: bool = False,
        baseline: float = 0.0,
        return_logits: bool = False,
    ) -> dict[int, float]:
        """Probabilities for many nodes under per-node feature interventions.

        ``masks`` maps a node to the feature indices to neutralise, or -- when
        ``keep_only`` is set -- the only feature indices to retain.
        """
        adj = self._adj_for(self.base_edge_index)
        out: dict[int, float] = {}
        for group in groups:
            x = self.x.clone()
            for node in group:
                idx = list(masks.get(node, []))
                if keep_only:
                    row = torch.full((x.size(1),), baseline, dtype=x.dtype)
                    if idx:
                        row[idx] = self.x[node][idx]
                    x[node] = row
                elif idx:
                    x[node, idx] = baseline
            with torch.no_grad():
                logits = self._forward(x, self.base_edge_index, adj)
            values = logits if return_logits else torch.sigmoid(logits)
            for node in group:
                out[node] = float(values[node].item())
        return out

    def batch_edge_removed_probabilities(
        self, nodes: Sequence[int], groups: list[list[int]], return_logits: bool = False
    ) -> dict[int, float]:
        """Probabilities after isolating each node from its payment links."""
        out: dict[int, float] = {}
        edges = self.base_edge_index
        not_self_loop = edges[0] != edges[1]
        for group in groups:
            member = torch.zeros(self.num_nodes, dtype=torch.bool)
            member[list(group)] = True
            incident = (member[edges[0]] | member[edges[1]]) & not_self_loop
            edge_index = edges[:, ~incident]
            adj = self._adj_for(edge_index)
            with torch.no_grad():
                logits = self._forward(self.x, edge_index, adj)
            values = logits if return_logits else torch.sigmoid(logits)
            for node in group:
                out[node] = float(values[node].item())
        return out

    def gradient_attribution(self, nodes: Sequence[int]) -> np.ndarray:
        """Gradient x input attribution of the illicit logit w.r.t. node features.

        Returns an array of shape ``(len(nodes), num_features)``.  A single
        backward pass per node is required because the readout is per node.
        """
        adj = self._adj_for(self.base_edge_index)
        out = np.zeros((len(nodes), self.x.size(1)), dtype=np.float32)
        for row, node in enumerate(nodes):
            x = self.x.clone().requires_grad_(True)
            logits = self._forward(x, self.base_edge_index, adj)
            self.model.zero_grad(set_to_none=True)
            logits[node].backward()
            grad = x.grad[node].detach()
            out[row] = (grad * self.x[node]).numpy()
        return out
