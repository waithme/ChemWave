from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


class TargetConditionedWaveletBlock(nn.Module):
    """Learns a target-specific spectral response to the same chemical graph signal."""

    def __init__(
        self, hidden_dim: int, bond_dim: int, num_targets: int, dropout: float
    ) -> None:
        super().__init__()
        self.edge_projection = nn.Linear(bond_dim, hidden_dim)
        self.neighbor_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.low_filter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.high_filter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.local_frequency_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.target_frequency = nn.Embedding(num_targets, hidden_dim)
        nn.init.zeros_(self.target_frequency.weight)
        self.normalization = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_target: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        message = self.neighbor_message(h[source] + self.edge_projection(edge_attr))
        neighbor_sum = scatter(message, target, dim=0, dim_size=h.shape[0], reduce="sum")
        degree = scatter(
            torch.ones(target.shape[0], dtype=h.dtype, device=h.device),
            target,
            dim=0,
            dim_size=h.shape[0],
            reduce="sum",
        ).unsqueeze(-1)
        low = (h + neighbor_sum) / (1.0 + degree)
        high = h - low
        gate_logits = self.local_frequency_gate(torch.cat([h, low, high], dim=-1))
        gate = torch.sigmoid(gate_logits + self.target_frequency(node_target))
        update = (1.0 - gate) * self.low_filter(low) + gate * self.high_filter(high)
        return self.dropout(torch.relu(self.normalization(h + update)))


class TargetConditionedChemWave(nn.Module):
    def __init__(
        self,
        num_targets: int,
        atom_dim: int = 35,
        relative_position_dim: int = 45,
        bond_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TargetConditionedWaveletBlock(
                    hidden_dim, bond_dim, num_targets, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.molecular_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.output_weight = nn.Parameter(torch.empty(num_targets, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_targets))
        nn.init.xavier_uniform_(self.output_weight)
        # Define the ablation-only parameters after every V1 parameter so the
        # same seed produces exactly the same initialization for all V1 keys.
        v1_rng_state = torch.random.get_rng_state()
        self.relative_position_projection = nn.Linear(
            relative_position_dim, hidden_dim, bias=False
        )
        self.target_relative_position = nn.Embedding(
            num_targets, relative_position_dim
        )
        nn.init.zeros_(self.relative_position_projection.weight)
        nn.init.zeros_(self.target_relative_position.weight)
        # The two constructors sample values before zero initialization. Undo
        # that RNG advance so V1 and this ablation also start with equal dropout
        # streams under the same seed.
        torch.random.set_rng_state(v1_rng_state)

    def forward(self, data: Batch) -> torch.Tensor:
        graph_target = data.target_id.view(-1)
        node_target = graph_target[data.batch]
        relative_gate = torch.sigmoid(
            self.target_relative_position(node_target)
        )
        relative_position = data.relative_position * relative_gate
        h = self.atom_projection(data.x) + self.relative_position_projection(
            relative_position
        )
        for block in self.blocks:
            h = block(h, data.edge_index, data.edge_attr, node_target)
        molecule = scatter(
            h, data.batch, dim=0, dim_size=data.num_graphs, reduce="mean"
        )
        molecule = self.molecular_transform(molecule)
        weight = self.output_weight[graph_target]
        return torch.sum(molecule * weight, dim=-1) + self.output_bias[graph_target]
