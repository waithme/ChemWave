from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


IDENTITY_COLUMNS = ["target", "row_id", "smiles", "cliff"]
MEASURE_COLUMNS = [
    "heavy_atom_count",
    "relative_total_abs",
    "relative_mean_abs_per_heavy_atom",
    "bond_total_abs",
    "bond_mean_abs_per_heavy_atom",
]


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    count = mol.GetNumHeavyAtoms()
    if count < 1:
        raise ValueError(f"Molecule has no heavy atoms: {smiles}")
    return count


def add_size_adjusted_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "seed",
        "relative_total_abs",
        "bond_total_abs",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    output = frame.copy()
    output["heavy_atom_count"] = output["smiles"].map(heavy_atom_count)
    output["relative_mean_abs_per_heavy_atom"] = (
        output["relative_total_abs"] / output["heavy_atom_count"]
    )
    output["bond_mean_abs_per_heavy_atom"] = (
        output["bond_total_abs"] / output["heavy_atom_count"]
    )
    numeric = output[MEASURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Size-adjusted intervention metrics contain non-finite values")
    return output


def collapse_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    seed_counts = frame.groupby(IDENTITY_COLUMNS, dropna=False)["seed"].nunique()
    if seed_counts.nunique() != 1:
        raise ValueError("Molecules do not have a common number of model seeds")
    return (
        frame.groupby(IDENTITY_COLUMNS, as_index=False, dropna=False)[
            MEASURE_COLUMNS
        ]
        .mean()
        .sort_values(["target", "row_id"])
        .reset_index(drop=True)
    )


def summarize_groups(molecules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cliff, group in molecules.groupby("cliff", sort=True):
        for metric in MEASURE_COLUMNS:
            values = group[metric]
            rows.append(
                {
                    "cliff": int(cliff),
                    "metric": metric,
                    "molecule_count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(values.median()),
                }
            )
    return pd.DataFrame(rows)


def target_contrasts(molecules: pd.DataFrame) -> pd.DataFrame:
    grouped = molecules.groupby(["target", "cliff"])[MEASURE_COLUMNS].mean()
    targets = sorted(molecules["target"].unique())
    rows = []
    for target in targets:
        if (target, 0) not in grouped.index or (target, 1) not in grouped.index:
            raise ValueError(f"Target lacks a cliff or non-cliff group: {target}")
        row = {"target": target}
        for metric in MEASURE_COLUMNS:
            noncliff = float(grouped.loc[(target, 0), metric])
            cliff = float(grouped.loc[(target, 1), metric])
            row[f"{metric}_noncliff"] = noncliff
            row[f"{metric}_cliff"] = cliff
            row[f"{metric}_cliff_minus_noncliff"] = cliff - noncliff
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute cliff/non-cliff intervention summaries after normalizing "
            "total influence by RDKit heavy-atom count."
        )
    )
    parser.add_argument("--molecule-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = add_size_adjusted_metrics(pd.read_csv(args.molecule_scores))
    molecules = collapse_seeds(frame)
    summary = summarize_groups(molecules)
    contrasts = target_contrasts(molecules)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "molecule_scores_size_adjusted.csv", index=False)
    molecules.to_csv(
        args.output_dir / "molecule_scores_five_seed_mean.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "cliff_noncliff_size_adjusted_summary.csv", index=False
    )
    contrasts.to_csv(
        args.output_dir / "cliff_noncliff_size_adjusted_by_target.csv", index=False
    )

    summary_wide = summary.pivot(index="metric", columns="cliff", values="mean")
    report = {
        "input": str(args.molecule_scores.resolve()),
        "molecule_count": int(len(molecules)),
        "targets": int(molecules["target"].nunique()),
        "seeds_per_molecule": int(frame["seed"].nunique()),
        "relative_total_percent_difference": float(
            100
            * (
                summary_wide.loc["relative_total_abs", 1]
                / summary_wide.loc["relative_total_abs", 0]
                - 1
            )
        ),
        "relative_per_heavy_atom_percent_difference": float(
            100
            * (
                summary_wide.loc["relative_mean_abs_per_heavy_atom", 1]
                / summary_wide.loc["relative_mean_abs_per_heavy_atom", 0]
                - 1
            )
        ),
        "targets_with_higher_relative_per_heavy_atom_in_cliffs": int(
            (
                contrasts[
                    "relative_mean_abs_per_heavy_atom_cliff_minus_noncliff"
                ]
                > 0
            ).sum()
        ),
        "interpretation": (
            "Descriptive size-normalized sensitivity analysis; not a test of "
            "causal mechanism or cliff-specific predictive improvement."
        ),
    }
    (args.output_dir / "size_adjusted_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
