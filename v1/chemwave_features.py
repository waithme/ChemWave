from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def _one_hot_other(value, choices) -> list[float]:
    return [float(value == choice) for choice in choices] + [
        float(value not in choices)
    ]


def atom35(mol: Chem.Mol) -> np.ndarray:
    features = []
    atomic_numbers = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    degrees = [0, 1, 2, 3, 4, 5]
    charges = [-2, -1, 0, 1, 2]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    for atom in mol.GetAtoms():
        row = _one_hot_other(atom.GetAtomicNum(), atomic_numbers)
        row += _one_hot_other(atom.GetTotalDegree(), degrees)
        row += _one_hot_other(atom.GetFormalCharge(), charges)
        row += _one_hot_other(atom.GetHybridization(), hybridizations)
        row += [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetTotalNumHs()),
            float(atom.GetMass() * 0.01),
        ]
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def bond12(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    edge_indices, edge_features = [], []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    for bond in mol.GetBonds():
        row = _one_hot_other(bond.GetBondType(), bond_types)
        row += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        row += _one_hot_other(bond.GetStereo(), stereos)
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend(((source, target), (target, source)))
        edge_features.extend((row, row))
    if not edge_indices:
        return np.empty((2, 0), dtype=np.int64), np.empty(
            (0, 12), dtype=np.float32
        )
    return np.asarray(edge_indices, dtype=np.int64).T, np.asarray(
        edge_features, dtype=np.float32
    )


def molecule_to_graph35(
    smiles: str,
    y: float,
    cliff_label: int | None = None,
    target_index: int = 0,
) -> Data:
    """Build an Atom35/Bond12 graph; cliff labels are test-only and optional."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    edge_index, edge_attr = bond12(mol)
    fields = {
        "x": torch.tensor(atom35(mol), dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "y": torch.tensor([y], dtype=torch.float32),
        "target_id": torch.tensor([target_index], dtype=torch.long),
        "smiles": smiles,
    }
    if cliff_label is not None:
        fields["cliff_label"] = torch.tensor([cliff_label], dtype=torch.long)
    return Data(**fields)
