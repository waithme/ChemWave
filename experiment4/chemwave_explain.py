from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter


@torch.no_grad()
def forward_with_interventions(
    model,
    data: Batch,
    *,
    relative_atom_mask: torch.Tensor | None = None,
    bond_gradient_mask: torch.Tensor | None = None,
    return_details: bool = False,
):
    """Run the unchanged trained model with explanation-only component masks.

    A relative mask removes only an atom's 45-dimensional relative-position
    channel. A bond mask removes only a directed edge from the proposed plain
    gradient; the ordinary ChemWave message path remains intact.
    """
    graph_target = data.target_id.view(-1)
    node_target = graph_target[data.batch]
    relative_gate = torch.sigmoid(model.target_relative_position(node_target))
    relative_position = data.relative_position * relative_gate
    if relative_atom_mask is not None:
        relative_position = relative_position * relative_atom_mask.view(-1, 1)
    relative_injection = model.relative_position_projection(relative_position)
    h = model.atom_projection(data.x) + relative_injection
    source, target = data.edge_index
    details = {
        "relative_gate": relative_gate,
        "relative_injection_norm": relative_injection.norm(dim=-1),
        "layers": [],
    }

    for block in model.blocks:
        message = block.neighbor_message(
            h[source] + block.edge_projection(data.edge_attr)
        )
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

        target_bond = block.target_bond_conductance(node_target[target])
        conductance_logits = block.edge_conductance(data.edge_attr) + torch.sum(
            target_bond * data.edge_attr, dim=-1, keepdim=True
        ) / math.sqrt(block.bond_dim)
        conductance = torch.sigmoid(conductance_logits)
        effective_conductance = conductance
        if bond_gradient_mask is not None:
            effective_conductance = conductance * bond_gradient_mask.view(-1, 1)
        edge_gradient = effective_conductance * (h[target] - h[source])
        gradient = scatter(
            edge_gradient, target, dim=0, dim_size=h.shape[0], reduce="sum"
        ) / degree.clamp_min(1.0)
        gradient_gain = torch.tanh(block.target_gradient_gain(node_target))
        high = (h - low) + gradient_gain * gradient

        gate_logits = block.local_frequency_gate(torch.cat([h, low, high], dim=-1))
        frequency_gate = torch.sigmoid(
            gate_logits + block.target_frequency(node_target)
        )
        update = (1.0 - frequency_gate) * block.low_filter(low)
        update = update + frequency_gate * block.high_filter(high)

        if return_details:
            normalized_edge_signal = (
                gradient_gain[target]
                * conductance
                * (h[target] - h[source])
                / degree[target].clamp_min(1.0)
            )
            details["layers"].append(
                {
                    "conductance": conductance.view(-1),
                    "gradient_norm": normalized_edge_signal.norm(dim=-1),
                    "frequency_gate_mean": frequency_gate.mean(dim=-1),
                }
            )
        h = block.dropout(torch.relu(block.normalization(h + update)))

    molecule = scatter(
        h, data.batch, dim=0, dim_size=data.num_graphs, reduce="mean"
    )
    molecule = model.molecular_transform(molecule)
    weight = model.output_weight[graph_target]
    prediction = torch.sum(molecule * weight, dim=-1) + model.output_bias[graph_target]
    return (prediction, details) if return_details else prediction


def undirected_edge_groups(edge_index: torch.Tensor) -> list[dict]:
    """Group the two directed PyG edges belonging to each chemical bond."""
    groups: dict[tuple[int, int], list[int]] = {}
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        key = tuple(sorted((int(source), int(target))))
        groups.setdefault(key, []).append(edge_id)
    return [
        {"atoms": key, "directed_edge_ids": edge_ids}
        for key, edge_ids in groups.items()
    ]


@torch.no_grad()
def _masked_predictions(
    model,
    graph: Data,
    *,
    relative_masks: Iterable[np.ndarray] | None = None,
    bond_masks: Iterable[np.ndarray] | None = None,
) -> np.ndarray:
    device = next(model.parameters()).device
    relative_masks = list(relative_masks or [])
    bond_masks = list(bond_masks or [])
    count = max(len(relative_masks), len(bond_masks))
    if count == 0:
        return np.empty(0, dtype=np.float32)
    if not relative_masks:
        relative_masks = [np.ones(graph.num_nodes, dtype=np.float32)] * count
    if not bond_masks:
        bond_masks = [np.ones(graph.num_edges, dtype=np.float32)] * count
    if len(relative_masks) != count or len(bond_masks) != count:
        raise ValueError("Relative and bond mask counts must match")
    batch = Batch.from_data_list([graph.clone() for _ in range(count)]).to(device)
    relative_mask = torch.from_numpy(np.concatenate(relative_masks)).to(device)
    bond_mask = torch.from_numpy(np.concatenate(bond_masks)).to(device)
    return (
        forward_with_interventions(
            model,
            batch,
            relative_atom_mask=relative_mask,
            bond_gradient_mask=bond_mask,
        )
        .detach()
        .cpu()
        .numpy()
    )


@torch.no_grad()
def explain_molecule(model, graph: Data) -> dict:
    """Return faithful atom/bond interventions and intrinsic model signals."""
    model.eval()
    device = next(model.parameters()).device
    batch = Batch.from_data_list([graph]).to(device)
    prediction, details = forward_with_interventions(
        model, batch, return_details=True
    )
    full_prediction = float(prediction.item())

    atom_masks = []
    for atom_index in range(graph.num_nodes):
        mask = np.ones(graph.num_nodes, dtype=np.float32)
        mask[atom_index] = 0.0
        atom_masks.append(mask)
    atom_predictions = _masked_predictions(
        model, graph, relative_masks=atom_masks
    )
    atom_delta = full_prediction - atom_predictions

    groups = undirected_edge_groups(graph.edge_index)
    bond_masks = []
    for group in groups:
        mask = np.ones(graph.num_edges, dtype=np.float32)
        mask[group["directed_edge_ids"]] = 0.0
        bond_masks.append(mask)
    bond_predictions = _masked_predictions(model, graph, bond_masks=bond_masks)
    bond_delta = full_prediction - bond_predictions

    layer_conductance = [
        layer["conductance"].detach().cpu().numpy() for layer in details["layers"]
    ]
    layer_gradient = [
        layer["gradient_norm"].detach().cpu().numpy() for layer in details["layers"]
    ]
    layer_frequency = [
        layer["frequency_gate_mean"].detach().cpu().numpy()
        for layer in details["layers"]
    ]
    for bond_index, group in enumerate(groups):
        directed = group["directed_edge_ids"]
        group["conductance_mean"] = float(
            np.mean([values[directed].mean() for values in layer_conductance])
        )
        group["gradient_norm_mean"] = float(
            np.mean([values[directed].mean() for values in layer_gradient])
        )
        group["prediction_delta"] = float(bond_delta[bond_index])

    effective_gate = (
        torch.sigmoid(model.target_relative_position.weight[graph.target_id.item()])
        * model.relative_position_projection.weight.norm(dim=0)
    )
    return {
        "prediction": full_prediction,
        "atom_prediction_delta": atom_delta.astype(np.float64),
        "atom_relative_injection_norm": details["relative_injection_norm"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "atom_frequency_gate_mean": np.mean(layer_frequency, axis=0).astype(
            np.float64
        ),
        "bonds": groups,
        "effective_relative_gate": effective_gate.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    }


@torch.no_grad()
def faithfulness_interventions(
    model,
    graph: Data,
    explanation: dict,
    *,
    top_k: tuple[int, ...] = (1, 3, 5),
    random_repeats: int = 5,
    random_seed: int = 0,
) -> list[dict]:
    """Compare top-ranked component removal with size-matched random removal."""
    rng = np.random.RandomState(random_seed)
    full_prediction = explanation["prediction"]
    records = []
    for component in ("relative_atom", "gradient_bond"):
        if component == "relative_atom":
            scores = np.abs(explanation["atom_prediction_delta"])
            size = graph.num_nodes
            groups = None
        else:
            scores = np.abs(
                np.asarray([bond["prediction_delta"] for bond in explanation["bonds"]])
            )
            size = len(explanation["bonds"])
            groups = explanation["bonds"]
        for requested_k in top_k:
            k = min(requested_k, size)
            if k == 0:
                continue
            top_indices = np.argsort(-scores)[:k]
            selections = [top_indices]
            selections.extend(
                rng.choice(size, size=k, replace=False)
                for _ in range(random_repeats)
            )
            relative_masks, bond_masks = [], []
            for selection in selections:
                relative_mask = np.ones(graph.num_nodes, dtype=np.float32)
                bond_mask = np.ones(graph.num_edges, dtype=np.float32)
                if component == "relative_atom":
                    relative_mask[selection] = 0.0
                else:
                    for bond_index in selection:
                        bond_mask[groups[int(bond_index)]["directed_edge_ids"]] = 0.0
                relative_masks.append(relative_mask)
                bond_masks.append(bond_mask)
            predictions = _masked_predictions(
                model,
                graph,
                relative_masks=relative_masks,
                bond_masks=bond_masks,
            )
            changes = np.abs(full_prediction - predictions)
            records.append(
                {
                    "component": component,
                    "k": int(k),
                    "top_change": float(changes[0]),
                    "random_change_mean": float(changes[1:].mean()),
                    "random_change_std": float(changes[1:].std(ddof=1))
                    if random_repeats > 1
                    else 0.0,
                }
            )
    return records
