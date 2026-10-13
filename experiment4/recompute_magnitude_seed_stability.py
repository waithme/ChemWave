"""Correct seed-stability ranks from saved component interventions, without inference.

Historical seed_stability.csv files may store signed-rank correlations in the
main columns; corrected files retain them in separately named diagnostic
columns. This script validates the signed diagnostics against the component
tables and writes absolute-magnitude rankings to a separate output file.
The input tables and their original audit files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from run_interpretability import signed_spearman, spearman


TABLES = Path(__file__).resolve().parent / "interpretability_outputs" / "tables"
SEEDS = (0, 1, 2, 3, 4)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_arrays(path: Path, index_column: str, value_column: str):
    frame = pd.read_csv(
        path,
        usecols=["target", "row_id", "seed", index_column, value_column],
    )
    if frame.duplicated(["target", "row_id", "seed", index_column]).any():
        raise ValueError(f"Duplicate component index in {path}")
    if not np.isfinite(frame[value_column].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite intervention value in {path}")
    grouped = defaultdict(dict)
    for (target, row_id, seed), rows in frame.groupby(
        ["target", "row_id", "seed"], sort=False
    ):
        rows = rows.sort_values(index_column)
        grouped[(target, int(row_id))][int(seed)] = (
            tuple(rows[index_column].astype(int)),
            rows[value_column].to_numpy(dtype=float),
        )
    return grouped


def pairwise_mean(arrays, metric):
    values = np.asarray([metric(left, right) for left, right in combinations(arrays, 2)])
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else np.nan


def checked_arrays(grouped, key, channel):
    if key not in grouped or tuple(sorted(grouped[key])) != SEEDS:
        raise ValueError(f"Missing five {channel} seeds for {key}")
    pairs = [grouped[key][seed] for seed in SEEDS]
    if len({indices for indices, _ in pairs}) != 1:
        raise ValueError(f"Inconsistent {channel} component indices for {key}")
    return [values for _, values in pairs]


def _write_new_or_identical(path: Path, contents: bytes) -> None:
    if path.exists():
        if path.read_bytes() != contents:
            raise RuntimeError(f"Refusing to overwrite a different existing output: {path}")
        return
    with path.open("xb") as stream:
        stream.write(contents)


def recompute_saved_stability(
    tables: Path, *, expected_molecules: int = 1476, expected_assays: int = 30
) -> dict:
    paths = {
        "atom": tables / "atom_scores.csv",
        "bond": tables / "bond_scores.csv",
        "stability": tables / "seed_stability.csv",
    }
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    atom = component_arrays(paths["atom"], "atom_index", "relative_prediction_delta")
    bond = component_arrays(paths["bond"], "bond_index", "gradient_prediction_delta")
    frame = pd.read_csv(paths["stability"])
    if len(frame) != expected_molecules or frame.target.nunique() != expected_assays:
        raise ValueError(
            f"Expected {expected_molecules} sampled molecules across {expected_assays} assays"
        )
    if frame.duplicated(["target", "row_id"]).any():
        raise ValueError("Duplicate molecule in seed_stability.csv")
    keys = list(zip(frame.target, frame.row_id.astype(int)))
    if set(keys) != set(atom) or set(keys) != set(bond):
        raise ValueError("Molecule keys differ between saved intervention tables")

    computed = defaultdict(list)
    for key in keys:
        for channel, grouped in (("atom", atom), ("bond", bond)):
            arrays = checked_arrays(grouped, key, channel)
            computed[f"{channel}_spearman_mean"].append(pairwise_mean(arrays, spearman))
            computed[f"{channel}_signed_spearman_mean"].append(
                pairwise_mean(arrays, signed_spearman)
            )

    for channel in ("atom", "bond"):
        signed = f"{channel}_signed_spearman_mean"
        historical = signed if signed in frame else f"{channel}_spearman_mean"
        if not np.allclose(frame[historical], computed[signed], rtol=0, atol=1e-10,
                           equal_nan=True):
            raise ValueError(f"Saved {historical} does not match the raw interventions")

    for column, values in computed.items():
        frame[column] = values
    output = tables / "seed_stability_magnitude.csv"
    _write_new_or_identical(output, frame.to_csv(index=False).encode("utf-8"))
    summary = {
        "definition": "Per-molecule mean over ten seed pairs of Spearman ranks on absolute single-component prediction deltas",
        "source_sha256": source_hashes,
        "input_table": paths["stability"].name,
        "output_table": output.name,
        "result_sha256": file_sha256(output),
        "molecules": len(frame),
        "assays": frame.target.nunique(),
        "seeds": list(SEEDS),
        "mean_over_molecules": {
            column: float(np.nanmean(frame[column])) for column in computed
        },
        "top3_columns_unchanged": ["atom_top3_overlap_mean", "bond_top3_overlap_mean"],
        "source_tables_unchanged": True,
        "new_inference": False,
    }
    if {name: file_sha256(path) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("An input table changed during re-aggregation")
    audit = tables / "seed_stability_magnitude_output_audit.json"
    _write_new_or_identical(audit, (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    summary = recompute_saved_stability(args.tables_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
