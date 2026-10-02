from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP


TARGET_EMBEDDING_DIM = 81
TARGET_ADAPTER_LAYERS = 3


class GINEBackbone(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolutions.append(GINEConv(mlp, edge_dim=bond_dim))
            self.normalizations.append(nn.BatchNorm1d(hidden_dim))
        self.graph_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        h = self.atom_projection(data.x)
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            update = convolution(h, data.edge_index, data.edge_attr)
            h = self.dropout(torch.relu(normalization(h + update)))
        pooled = global_mean_pool(h, data.batch, size=data.num_graphs)
        return self.graph_transform(pooled)


class AttentiveFPBackbone(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=atom_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=bond_dim,
            num_layers=num_layers,
            num_timesteps=2,
            dropout=dropout,
        )

    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch)


class MatchedMultiTargetModel(nn.Module):
    """Shared molecular backbone with exactly 2,182 active target coordinates.

    The two matched baselines use the same target adapter. Per target it contains
    three 300-dimensional FiLM scale/shift pairs (1,800 coordinates), an
    81-dimensional target embedding, and a 300-dimensional linear output head
    plus bias (301 coordinates). The chemical relative-position field is never
    read by either backbone.
    """

    def __init__(
        self,
        model_name: str,
        num_targets: int,
        atom_dim: int = 35,
        bond_dim: int = 12,
        hidden_dim: int = 300,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim != 300:
            raise ValueError(
                "Matched adaptation budget is defined for hidden_dim=300"
            )
        if num_layers != TARGET_ADAPTER_LAYERS:
            raise ValueError("Matched adapter requires exactly three layers")
        self.model_name = model_name
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        if model_name == "gine":
            self.backbone = GINEBackbone(
                atom_dim, bond_dim, hidden_dim, num_layers, dropout
            )
        elif model_name == "attentivefp":
            self.backbone = AttentiveFPBackbone(
                atom_dim, bond_dim, hidden_dim, num_layers, dropout
            )
        else:
            raise ValueError(f"Unknown matched baseline: {model_name}")

        self.target_embedding = nn.Embedding(num_targets, TARGET_EMBEDDING_DIM)
        self.target_projection = nn.Linear(
            TARGET_EMBEDDING_DIM, hidden_dim, bias=False
        )
        self.adapter_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.adapter_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.adapter_dropout = nn.Dropout(dropout)
        self.target_scale = nn.Embedding(num_targets, num_layers * hidden_dim)
        self.target_shift = nn.Embedding(num_targets, num_layers * hidden_dim)
        self.output_weight = nn.Parameter(torch.empty(num_targets, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_targets))

        nn.init.zeros_(self.target_embedding.weight)
        nn.init.zeros_(self.target_scale.weight)
        nn.init.zeros_(self.target_shift.weight)
        nn.init.xavier_uniform_(self.output_weight)

        if self.active_target_coordinate_count() != 2182:
            raise AssertionError("Matched target-adaptation budget is not 2,182")

    def target_row_parameters(self) -> list[nn.Parameter]:
        return [
            self.target_embedding.weight,
            self.target_scale.weight,
            self.target_shift.weight,
            self.output_weight,
            self.output_bias,
        ]

    def active_target_coordinate_count(self) -> int:
        return sum(parameter.shape[1:].numel() for parameter in self.target_row_parameters())

    def freeze_shared_modules_eval(self) -> None:
        self.backbone.eval()
        self.target_projection.eval()
        self.adapter_layers.eval()
        self.adapter_norms.eval()
        self.adapter_dropout.eval()

    def forward(self, data: Batch) -> torch.Tensor:
        graph_target = data.target_id.view(-1)
        molecule = self.backbone(data)
        molecule = molecule + self.target_projection(
            self.target_embedding(graph_target)
        )
        scale = self.target_scale(graph_target).view(
            -1, TARGET_ADAPTER_LAYERS, self.hidden_dim
        )
        shift = self.target_shift(graph_target).view(
            -1, TARGET_ADAPTER_LAYERS, self.hidden_dim
        )
        for layer_index, (layer, normalization) in enumerate(
            zip(self.adapter_layers, self.adapter_norms)
        ):
            candidate = torch.relu(layer(molecule))
            adapted = (
                (1.0 + torch.tanh(scale[:, layer_index])) * candidate
                + shift[:, layer_index]
            )
            molecule = normalization(
                molecule + self.adapter_dropout(adapted)
            )
        weight = self.output_weight[graph_target]
        return torch.sum(molecule * weight, dim=-1) + self.output_bias[graph_target]


def build_model(
    model_name: str,
    num_targets: int,
    hidden_dim: int = 300,
) -> MatchedMultiTargetModel:
    return MatchedMultiTargetModel(
        model_name=model_name,
        num_targets=num_targets,
        hidden_dim=hidden_dim,
    )
