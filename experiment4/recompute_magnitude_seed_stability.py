"""Correct seed-stability ranks from saved component interventions, without inference.

The existing seed_stability.csv was computed from signed deltas.  This script
checks those historical values against atom_scores.csv and bond_scores.csv,
then makes the main Spearman columns rank absolute intervention magnitudes.
The signed diagnostic is retained under explicitly named columns.
"""

from __future__ import annotations

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


def main() -> None:
    paths = {
        "atom": TABLES / "atom_scores.csv",
        "bond": TABLES / "bond_scores.csv",
        "stability": TABLES / "seed_stability.csv",
    }
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    atom = component_arrays(paths["atom"], "atom_index", "relative_prediction_delta")
    bond = component_arrays(paths["bond"], "bond_index", "gradient_prediction_delta")
    frame = pd.read_csv(paths["stability"])
    if len(frame) != 1476 or frame.target.nunique() != 30:
        raise ValueError("Expected 1,476 sampled molecules across 30 assays")
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
    frame.to_csv(paths["stability"], index=False)
    summary = {
        "definition": "Per-molecule mean over ten seed pairs of Spearman ranks on absolute single-component prediction deltas",
        "source_sha256_before_update": source_hashes,
        "result_sha256": file_sha256(paths["stability"]),
        "molecules": len(frame),
        "assays": frame.target.nunique(),
        "seeds": list(SEEDS),
        "mean_over_molecules": {
            column: float(np.nanmean(frame[column])) for column in computed
        },
        "top3_columns_unchanged": ["atom_top3_overlap_mean", "bond_top3_overlap_mean"],
    }
    audit = TABLES / "seed_stability_magnitude_audit.json"
    audit.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
