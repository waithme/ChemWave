from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


VARIANT_CONFIGS = {
    name: {
        "use_relative": True,
        "target_relative": True,
        "use_bond_gradient": True,
        "use_bond_transport": False,
        "target_bond": True,
    }
    for name in ("exact_shell", "cumulative_shell")
}
VARIANT_NAMES = tuple(VARIANT_CONFIGS)


def get_variant_config(variant: str) -> dict[str, bool]:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown ablation variant {variant!r}; choose from {VARIANT_NAMES}"
        )
    return dict(VARIANT_CONFIGS[variant])


class TargetConditionedWaveletBlock(nn.Module):
    """Target-conditioned ChemWave block with an optional chemical gradient."""

    def __init__(
        self,
        hidden_dim: int,
        bond_dim: int,
        num_targets: int,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__()
        self.config = get_variant_config(variant)
        self.bond_dim = bond_dim
        self.edge_projection = nn.Linear(bond_dim, hidden_dim)
        self.neighbor_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.low_filter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.high_filter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_frequency_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.target_frequency = nn.Embedding(num_targets, hidden_dim)
        nn.init.zeros_(self.target_frequency.weight)
        self.normalization = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Ablation-only components are created after all V1 components. Their
        # zero initialization and RNG restoration preserve a fair V1 start.
        v1_rng_state = torch.random.get_rng_state()
        if self.config["use_bond_gradient"]:
            if self.config["use_bond_transport"]:
                self.bond_transport_scale = nn.Linear(
                    bond_dim, hidden_dim, bias=False
                )
                self.bond_transport_shift = nn.Linear(
                    bond_dim, hidden_dim, bias=False
                )
                nn.init.zeros_(self.bond_transport_scale.weight)
                nn.init.zeros_(self.bond_transport_shift.weight)
            self.edge_conductance = nn.Linear(bond_dim, 1, bias=False)
            nn.init.zeros_(self.edge_conductance.weight)
            if self.config["target_bond"]:
                self.target_bond_conductance = nn.Embedding(
                    num_targets, bond_dim
                )
                self.target_gradient_gain = nn.Embedding(
                    num_targets, hidden_dim
                )
                nn.init.zeros_(self.target_bond_conductance.weight)
                nn.init.zeros_(self.target_gradient_gain.weight)
            else:
                self.shared_gradient_gain = nn.Parameter(torch.zeros(hidden_dim))
        torch.random.set_rng_state(v1_rng_state)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_target: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        message = self.neighbor_message(h[source] + self.edge_projection(edge_attr))
        neighbor_sum = scatter(
            message, target, dim=0, dim_size=h.shape[0], reduce="sum"
        )
        degree = scatter(
            torch.ones(target.shape[0], dtype=h.dtype, device=h.device),
            target,
            dim=0,
            dim_size=h.shape[0],
            reduce="sum",
        ).unsqueeze(-1)
        low = (h + neighbor_sum) / (1.0 + degree)
        high = h - low

        if self.config["use_bond_gradient"]:
            transported_neighbor = h[source]
            if self.config["use_bond_transport"]:
                scale = torch.tanh(self.bond_transport_scale(edge_attr))
                shift = self.bond_transport_shift(edge_attr)
                transported_neighbor = transported_neighbor * (1.0 + scale) + shift

            conductance_logits = self.edge_conductance(edge_attr)
            if self.config["target_bond"]:
                target_bond = self.target_bond_conductance(node_target[target])
                conductance_logits = conductance_logits + torch.sum(
                    target_bond * edge_attr, dim=-1, keepdim=True
                ) / math.sqrt(self.bond_dim)
                gradient_gain = torch.tanh(
                    self.target_gradient_gain(node_target)
                )
            else:
                gradient_gain = torch.tanh(self.shared_gradient_gain).unsqueeze(0)
            conductance = torch.sigmoid(conductance_logits)
            edge_gradient = conductance * (h[target] - transported_neighbor)
            gradient = scatter(
                edge_gradient, target, dim=0, dim_size=h.shape[0], reduce="sum"
            ) / degree.clamp_min(1.0)
            high = high + gradient_gain * gradient

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
        variant: str = "exact_shell",
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.variant = variant
        self.config = get_variant_config(variant)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TargetConditionedWaveletBlock(
                    hidden_dim, bond_dim, num_targets, dropout, variant
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

        v1_rng_state = torch.random.get_rng_state()
        if self.config["use_relative"]:
            self.relative_position_projection = nn.Linear(
                relative_position_dim, hidden_dim, bias=False
            )
            nn.init.zeros_(self.relative_position_projection.weight)
            if self.config["target_relative"]:
                self.target_relative_position = nn.Embedding(
                    num_targets, relative_position_dim
                )
                nn.init.zeros_(self.target_relative_position.weight)
        torch.random.set_rng_state(v1_rng_state)

    def forward(self, data: Batch) -> torch.Tensor:
        graph_target = data.target_id.view(-1)
        node_target = graph_target[data.batch]
        h = self.atom_projection(data.x)
        if self.config["use_relative"]:
            relative_position = (
                data.relative_position_cumulative
                if self.variant == "cumulative_shell"
                else data.relative_position
            )
            if self.config["target_relative"]:
                relative_position = relative_position * torch.sigmoid(
                    self.target_relative_position(node_target)
                )
            h = h + self.relative_position_projection(relative_position)
        for block in self.blocks:
            h = block(h, data.edge_index, data.edge_attr, node_target)
        molecule = scatter(
            h, data.batch, dim=0, dim_size=data.num_graphs, reduce="mean"
        )
        molecule = self.molecular_transform(molecule)
        weight = self.output_weight[graph_target]
        return torch.sum(molecule * weight, dim=-1) + self.output_bias[graph_target]
