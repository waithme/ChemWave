"""Regenerate the SI molecule gallery from saved five-seed intervention tables.

This does not load checkpoints or run the model. It preserves the 120 selection
records in molecule_gallery.tex, including the commented record moved to the
main paper. Depiction, annotation, and legend changes use the original saved
component deltas without changing molecular graphs or case selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from interpretability_visuals import draw_molecule_explanation


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "experiment4/interpretability_outputs/tables"
GALLERY_TEX = ROOT / "chemwave_supplementary_draft/tables/molecule_gallery.tex"
DEFAULT_OUTPUT = ROOT / "experiment4/interpretability_outputs/figures/molecule_cases"
SEEDS = {0, 1, 2, 3, 4}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def gallery_cases(gallery_tex: Path) -> list[tuple[str, int, str, Path]]:
    if not gallery_tex.is_file():
        raise FileNotFoundError(
            "The gallery-selection TeX is not included in the code-only repository; "
            "supply the original local selection with --gallery-tex."
        )
    text = gallery_tex.read_text()
    paths = re.findall(r"figures/molecule_cases/([^}]+\.png)", text)
    cases = []
    for escaped in paths:
        relative = Path(escaped.replace(r"\_", "_"))
        match = re.fullmatch(r"row_(\d+)_(.+)\.png", relative.name)
        if match is None or len(relative.parts) != 2:
            raise ValueError(f"Unexpected gallery path: {relative}")
        cases.append((relative.parent.name, int(match.group(1)), match.group(2), relative))
    if len(cases) != 120 or len({case[3] for case in cases}) != 120:
        raise ValueError("Expected 120 unique gallery panels")
    if len({case[0] for case in cases}) != 30:
        raise ValueError("Expected 30 gallery assays")
    return cases


def selected_groups(path: Path, columns: list[str], keys: pd.DataFrame):
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.merge(keys, on=["target", "row_id"], how="inner", validate="many_to_one")
    return {key: part for key, part in frame.groupby(["target", "row_id"], sort=False)}


def five_seed_values(frame: pd.DataFrame, index: str, value: str) -> pd.Series:
    if frame.duplicated(["seed", index]).any():
        raise ValueError(f"Duplicate {index} within a seed")
    pivot = frame.pivot(index=index, columns="seed", values=value).sort_index()
    if set(pivot.columns) != SEEDS or pivot.isna().any().any():
        raise ValueError(f"Incomplete five-seed {index} effects")
    if not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite five-seed {index} effects")
    if list(pivot.index) != list(range(len(pivot))):
        raise ValueError(f"Noncontiguous {index} indices")
    return pivot.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Render one assay for visual QA")
    parser.add_argument("--gallery-tex", type=Path, default=GALLERY_TEX)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    cases = gallery_cases(args.gallery_tex)
    if args.target:
        cases = [case for case in cases if case[0] == args.target]
        if not cases:
            raise ValueError(f"Unknown gallery assay: {args.target}")
    keys = pd.DataFrame(sorted({(target, row_id) for target, row_id, _, _ in cases}),
                        columns=["target", "row_id"])
    paths = {
        "molecule": args.tables_dir / "molecule_scores.csv",
        "atom": args.tables_dir / "atom_scores.csv",
        "bond": args.tables_dir / "bond_scores.csv",
    }
    molecules = selected_groups(
        paths["molecule"], ["target", "row_id", "smiles", "y", "cliff", "seed", "prediction"], keys
    )
    atoms = selected_groups(
        paths["atom"], ["target", "row_id", "seed", "atom_index", "relative_prediction_delta"], keys
    )
    bonds = selected_groups(
        paths["bond"], ["target", "row_id", "seed", "bond_index", "atom_i", "atom_j", "gradient_prediction_delta"], keys
    )
    expected_keys = set(map(tuple, keys.to_numpy()))
    if set(molecules) != expected_keys or set(atoms) != expected_keys or set(bonds) != expected_keys:
        raise ValueError("A selected case is missing from a saved result table")

    examples = {}
    for target, row_id, label, relative in cases:
        key = (target, row_id)
        molecule = molecules[key]
        if len(molecule) != 5 or set(molecule.seed) != SEEDS:
            raise ValueError(f"Expected five prediction seeds for {key}")
        for column in ("smiles", "y", "cliff"):
            if molecule[column].nunique() != 1:
                raise ValueError(f"Inconsistent {column} across seeds for {key}")
        smiles = str(molecule.smiles.iloc[0])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {key}")
        atom_means = five_seed_values(atoms[key], "atom_index", "relative_prediction_delta")
        bond_frame = bonds[key]
        bond_means = five_seed_values(bond_frame, "bond_index", "gradient_prediction_delta")
        if len(atom_means) != mol.GetNumAtoms() or len(bond_means) != mol.GetNumBonds():
            raise ValueError(f"Component counts do not match RDKit graph for {key}")
        metadata = bond_frame.groupby("bond_index")[["atom_i", "atom_j"]]
        if not metadata.nunique().eq(1).all().all():
            raise ValueError(f"Bond endpoints differ across seeds for {key}")
        endpoints = metadata.first().sort_index()
        bond_records = [
            {"atoms": (int(endpoints.loc[i, "atom_i"]), int(endpoints.loc[i, "atom_j"])),
             "prediction_delta": float(bond_means.loc[i])}
            for i in bond_means.index
        ]
        display_label = {
            "lowest_error": "sampled low error",
            "median_error": "sampled median error",
            "highest_error": "sampled high error",
            "multiple_halogen": "sampled halogen case",
        }.get(label)
        if display_label is None:
            raise ValueError(f"Unknown case label: {label}")
        mean_prediction = float(molecule.prediction.mean())
        legend = (
            f"{target} | {display_label} | cliff={int(molecule.cliff.iloc[0])}\n"
            f"y={float(molecule.y.iloc[0]):.3f}; mean pred (5 seeds)={mean_prediction:.3f}\n"
            "red raises / blue lowers (full-minus-masked)"
        )
        draw_molecule_explanation(
            smiles, atom_means.to_numpy(), bond_records,
            args.output_dir / relative, legend=legend,
            width=args.width, height=args.height,
            max_atom_annotations=0, show_atom_indices=False,
        )
        if not examples:
            examples[str(relative)] = {
                "mean_prediction": mean_prediction,
                "max_abs_atom_effect": float(atom_means.abs().max()),
                "max_abs_bond_effect": float(bond_means.abs().max()),
            }

    summary = {
        "rendering_rdkit_version": rdBase.rdkitVersion,
        "source_sha256": {name: digest(path) for name, path in paths.items()},
        "gallery_tex_sha256": digest(args.gallery_tex),
        "molecule_canvas_pixels": [args.width, args.height],
        "target_filter": args.target,
        "rendered_panels": len(cases),
        "assays": len({case[0] for case in cases}),
        "effect_definition": "Five-seed mean of signed full-minus-single-component-masked predictions",
        "color_scales": "Separate per-molecule maxima for atom and bond effects, printed on each panel",
        "example": examples,
    }
    if args.target is None:
        manifest = args.output_dir.parent / "case_gallery_regeneration_manifest.json"
        manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
