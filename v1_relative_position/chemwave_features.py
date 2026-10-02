from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Lipinski
from torch_geometric.data import Data


CHEMICAL_ROLE_NAMES = (
    "donor",
    "acceptor",
    "positive_ionizable",
    "negative_ionizable",
    "aromatic",
    "hydrophobe",
    "halogen",
    "ring_junction",
    "rotatable_connector",
)
RELATIVE_POSITION_RADII = (1, 2, 3, 4, 5)
RELATIVE_POSITION_DIM = len(CHEMICAL_ROLE_NAMES) * len(RELATIVE_POSITION_RADII)

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


def chemical_roles(mol: Chem.Mol) -> np.ndarray:
    """Return label-independent atom roles used as molecular position anchors."""
    roles = np.zeros(
        (mol.GetNumAtoms(), len(CHEMICAL_ROLE_NAMES)), dtype=np.float32
    )
    for match in mol.GetSubstructMatches(Lipinski.HDonorSmarts):
        roles[match[0], 0] = 1.0
    for match in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts):
        roles[match[0], 1] = 1.0

    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        if formal_charge > 0 or (
            atomic_number == 7
            and formal_charge == 0
            and not atom.GetIsAromatic()
            and atom.GetTotalDegree() <= 3
            and not _resonance_deactivated_nitrogen(atom)
        ):
            roles[atom_index, 2] = 1.0
        if formal_charge < 0 or _acidic_heteroatom(atom):
            roles[atom_index, 3] = 1.0
        if atom.GetIsAromatic():
            roles[atom_index, 4] = 1.0
        if (
            atomic_number == 6
            and formal_charge == 0
            and all(
                neighbor.GetAtomicNum() in {1, 6, 9, 17, 35, 53}
                for neighbor in atom.GetNeighbors()
            )
        ):
            roles[atom_index, 5] = 1.0
        if atomic_number in {9, 17, 35, 53}:
            roles[atom_index, 6] = 1.0
        if ring_info.NumAtomRings(atom_index) > 1:
            roles[atom_index, 7] = 1.0
    for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        roles[list(match), 8] = 1.0
    return roles


def _resonance_deactivated_nitrogen(atom: Chem.Atom) -> bool:
    """Detect amide/sulfonamide-like neutral nitrogens."""
    for neighbor in atom.GetNeighbors():
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if other.GetIdx() == atom.GetIdx():
                continue
            if (
                bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and other.GetAtomicNum() in {7, 8, 16}
            ):
                return True
    return False


def _acidic_heteroatom(atom: Chem.Atom) -> bool:
    """Detect neutral acidic O/S atoms attached to a resonance center."""
    if atom.GetAtomicNum() not in {8, 16} or atom.GetTotalNumHs() == 0:
        return False
    for center in atom.GetNeighbors():
        if center.GetAtomicNum() not in {6, 15, 16}:
            continue
        for bond in center.GetBonds():
            other = bond.GetOtherAtom(center)
            if other.GetIdx() == atom.GetIdx():
                continue
            if (
                bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and other.GetAtomicNum() in {8, 16}
            ):
                return True
    return False


def chemical_relative_position_field(mol: Chem.Mol) -> np.ndarray:
    """Encode chemical-role distributions in exact 1--5 hop shells.

    A shell is normalized by the square root of its atom count. This limits
    molecule-size effects without discarding whether a shell contains one or
    several atoms with the same role.
    """
    roles = chemical_roles(mol)
    distance = Chem.GetDistanceMatrix(mol)
    shells = []
    for radius in RELATIVE_POSITION_RADII:
        membership = (distance == radius).astype(np.float32)
        shell_size = membership.sum(axis=1, keepdims=True)
        normalization = np.sqrt(np.maximum(shell_size, 1.0))
        shells.append((membership @ roles) / normalization)
    return np.concatenate(shells, axis=1).astype(np.float32, copy=False)


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
        "relative_position": torch.tensor(
            chemical_relative_position_field(mol), dtype=torch.float32
        ),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "y": torch.tensor([y], dtype=torch.float32),
        "target_id": torch.tensor([target_index], dtype=torch.long),
        "smiles": smiles,
    }
    if cliff_label is not None:
        fields["cliff_label"] = torch.tensor([cliff_label], dtype=torch.long)
    return Data(**fields)
