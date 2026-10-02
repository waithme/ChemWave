from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chemwave_features import CHEMICAL_ROLE_NAMES, RELATIVE_POSITION_RADII


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Statistical figure generation requires matplotlib; "
            "install requirements.txt first."
        ) from error
    return plt


def _signed_color(value: float, scale: float) -> tuple[float, float, float]:
    strength = min(abs(value) / max(scale, 1e-12), 1.0) * 0.85
    if value >= 0:
        return (1.0, 1.0 - strength, 1.0 - strength)
    return (1.0 - strength, 1.0 - strength, 1.0)


def draw_molecule_explanation(
    smiles: str,
    atom_scores: np.ndarray,
    bonds: list[dict],
    output_path: Path,
    *,
    legend: str,
    width: int = 900,
    height: int = 600,
) -> None:
    """Draw relative-position atom and bond-difference intervention effects."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol)
    atom_scale = float(np.max(np.abs(atom_scores))) if len(atom_scores) else 1.0
    bond_values = np.asarray([bond["prediction_delta"] for bond in bonds])
    bond_scale = float(np.max(np.abs(bond_values))) if len(bond_values) else 1.0
    atom_colors = {
        index: _signed_color(float(score), atom_scale)
        for index, score in enumerate(atom_scores)
        if abs(score) > 1e-12
    }
    atom_radii = {
        index: 0.25 + 0.25 * min(abs(float(score)) / max(atom_scale, 1e-12), 1.0)
        for index, score in enumerate(atom_scores)
        if abs(score) > 1e-12
    }
    bond_colors = {}
    highlight_bonds = []
    for bond_record in bonds:
        begin, end = bond_record["atoms"]
        bond = mol.GetBondBetweenAtoms(int(begin), int(end))
        if bond is None:
            continue
        bond_id = bond.GetIdx()
        highlight_bonds.append(bond_id)
        bond_colors[bond_id] = _signed_color(
            float(bond_record["prediction_delta"]), bond_scale
        )

    top_atoms = np.argsort(-np.abs(atom_scores))[: min(5, len(atom_scores))]
    for atom_index in top_atoms:
        mol.GetAtomWithIdx(int(atom_index)).SetProp(
            "atomNote", f"{atom_scores[atom_index]:+.3f}"
        )
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.annotationFontScale = 0.7
    drawer.DrawMolecule(
        mol,
        legend=legend,
        highlightAtoms=list(atom_colors),
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(drawer.GetDrawingText())


def plot_target_gate(gate: np.ndarray, target: str, output_path: Path) -> None:
    plt = _pyplot()
    matrix = np.asarray(gate).reshape(
        len(RELATIVE_POSITION_RADII), len(CHEMICAL_ROLE_NAMES)
    ).T
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_title(f"Effective chemical relative-position response: {target}")
    axis.set_xlabel("Topological shell (hop)")
    axis.set_ylabel("Chemical role")
    axis.set_xticks(range(len(RELATIVE_POSITION_RADII)), RELATIVE_POSITION_RADII)
    axis.set_yticks(range(len(CHEMICAL_ROLE_NAMES)), CHEMICAL_ROLE_NAMES)
    fig.colorbar(image, ax=axis, label="gate × projection-column norm")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_all_target_gates(gates: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    pivot = gates.pivot_table(
        index="target", columns="dimension", values="effective_gate", aggfunc="mean"
    ).sort_index()
    fig, axis = plt.subplots(figsize=(17.0, 10.5))
    image = axis.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
    labels = []
    for radius in RELATIVE_POSITION_RADII:
        labels.extend(f"{role}@{radius}" for role in CHEMICAL_ROLE_NAMES)
    tick_positions = np.arange(len(labels))
    axis.set_xticks(tick_positions, [labels[index] for index in tick_positions])
    axis.tick_params(axis="x", labelrotation=90, labelsize=7.5)
    for boundary in range(len(CHEMICAL_ROLE_NAMES), len(labels), len(CHEMICAL_ROLE_NAMES)):
        axis.axvline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.8)
    axis.set_xlabel("Chemical role and topological shell")
    axis.set_ylabel("Target")
    axis.set_title("Effective relative-position response across targets")
    fig.colorbar(image, ax=axis, label="gate × projection-column norm")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cliff_noncliff(molecules: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    metrics = [
        ("relative_max_abs", "Max relative-atom intervention"),
        ("bond_max_abs", "Max bond-difference intervention"),
        ("prediction_abs_error", "Absolute prediction error"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 4.2))
    for axis, (column, title) in zip(axes, metrics):
        values = [
            molecules.loc[molecules["cliff"] == label, column].dropna().to_numpy()
            for label in (0, 1)
        ]
        axis.boxplot(values, tick_labels=["non-cliff", "cliff"], showfliers=False)
        axis.set_title(title)
        axis.set_ylabel("Absolute prediction change" if "error" not in column else "Error")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_faithfulness(records: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    summary = (
        records.groupby(["component", "k"])[["top_change", "random_change_mean"]]
        .mean()
        .reset_index()
    )
    components = list(summary["component"].unique())
    fig, axes = plt.subplots(1, len(components), figsize=(5.2 * len(components), 4.2))
    axes = np.atleast_1d(axes)
    for axis, component in zip(axes, components):
        frame = summary.loc[summary["component"] == component]
        positions = np.arange(len(frame))
        axis.bar(positions - 0.18, frame["top_change"], 0.36, label="Top-ranked")
        axis.bar(
            positions + 0.18,
            frame["random_change_mean"],
            0.36,
            label="Random",
        )
        axis.set_xticks(positions, [str(value) for value in frame["k"]])
        axis.set_xlabel("Number of components removed")
        axis.set_ylabel("Absolute prediction change")
        display_name = {
            "gradient_bond": "Bond-difference intervention",
            "relative_atom": "Relative-position atom",
        }.get(component, component.replace("_", " "))
        axis.set_title(display_name)
        axis.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_stability(stability: pd.DataFrame, output_path: Path) -> None:
    plt = _pyplot()
    columns = [
        "atom_spearman_mean",
        "atom_top3_overlap_mean",
        "bond_spearman_mean",
        "bond_top3_overlap_mean",
    ]
    labels = ["Atom Spearman", "Atom Top-3", "Bond Spearman", "Bond Top-3"]
    values = [stability[column].dropna().to_numpy() for column in columns]
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Across-seed agreement")
    axis.set_title("Intervention-ranking stability across five seeds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
