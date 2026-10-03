from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from chemwave_explain import explain_molecule, faithfulness_interventions
from chemwave_features import (
    CHEMICAL_ROLE_NAMES,
    RELATIVE_POSITION_RADII,
    molecule_to_graph35,
)
from chemwave_multitask import TargetConditionedChemWave
from interpretability_visuals import (
    draw_molecule_explanation,
    plot_all_target_gates,
    plot_cliff_noncliff,
    plot_faithfulness,
    plot_stability,
    plot_target_gate,
)

# The provenance implementation is shared with the final-model runner while
# this analysis keeps its original local model/feature modules.
sys.path.append(str(Path(__file__).resolve().parents[1] / "final_model"))
from chemwave_provenance import (  # noqa: E402
    MODEL_OPTIONS, VARIANT, canonical_sha256, sha256_file,
)
from chemwave_training import prepare_development_data  # noqa: E402


def load_model(
    checkpoint_path: Path,
    *,
    target: str,
    seed: int,
    device: torch.device,
    expected_fingerprint: str,
    expected_target_names: list[str],
    legacy_sha256: str | None = None,
) -> tuple[TargetConditionedChemWave, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected = {
        "stage": "adapted", "target_name": target, "seed": seed,
        "variant": VARIANT,
        "development_data_sha256": expected_fingerprint,
        "target_names": expected_target_names,
    }
    mismatches = {
        key: {"expected": value, "found": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint metadata mismatch for {checkpoint_path}:\n"
            + json.dumps(mismatches, indent=2)
        )
    if "protocol" in checkpoint or "protocol_sha256" in checkpoint:
        stored = checkpoint.get("protocol")
        if not isinstance(stored, dict) or checkpoint.get("protocol_sha256") != canonical_sha256(stored):
            raise RuntimeError(f"Corrupt checkpoint protocol: {checkpoint_path}")
        if stored.get("stage") != "adapted" or stored.get("variant") != VARIANT:
            raise RuntimeError(f"Wrong checkpoint protocol: {checkpoint_path}")
        state = checkpoint["model_state"]
        num_targets, hidden_dim = state["output_weight"].shape
        expected_model = {**MODEL_OPTIONS, "hidden_dim": hidden_dim, "num_targets": num_targets}
        if stored.get("model") != expected_model:
            raise RuntimeError(f"Checkpoint architecture differs from analysis model: {checkpoint_path}")
        for name in ("chemwave_features.py", "chemwave_multitask.py"):
            if stored.get("source_sha256", {}).get(name) != sha256_file(Path(__file__).parent / name):
                raise RuntimeError(f"Analysis source differs from checkpoint protocol: {name}")
    elif legacy_sha256 is None or sha256_file(checkpoint_path) != legacy_sha256:
        raise RuntimeError(
            f"Legacy checkpoint requires a matching read-only audit manifest: {checkpoint_path}"
        )
    state = checkpoint["model_state"]
    num_targets, hidden_dim = state["output_weight"].shape
    model = TargetConditionedChemWave(
        num_targets=num_targets, hidden_dim=hidden_dim
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def stratified_sample(
    frame: pd.DataFrame, max_cliff: int, max_noncliff: int
) -> pd.DataFrame:
    selected = []
    for cliff, maximum in ((1, max_cliff), (0, max_noncliff)):
        group = frame.loc[frame["cliff_mol"].astype(int) == cliff]
        if maximum > 0 and len(group) > maximum:
            group = group.sample(n=maximum, random_state=0)
        selected.append(group)
    return pd.concat(selected).sort_values("row_id").reset_index(drop=True)


def concentration(values: np.ndarray, k: int = 3) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    return float(np.sort(values)[-min(k, len(values)) :].sum() / total)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    value = left_rank.corr(right_rank)
    return float(value) if pd.notna(value) else np.nan


def top_k_overlap(left: np.ndarray, right: np.ndarray, k: int = 3) -> float:
    k = min(k, len(left), len(right))
    if k == 0:
        return np.nan
    left_top = set(np.argsort(-np.abs(left))[:k])
    right_top = set(np.argsort(-np.abs(right))[:k])
    return len(left_top & right_top) / k


def mean_pairwise(values: list[np.ndarray], metric) -> float:
    scores = [metric(left, right) for left, right in combinations(values, 2)]
    return float(np.nanmean(scores)) if scores else np.nan


def tidy_group_summary(
    frame: pd.DataFrame, group_columns: list[str], metrics: list[str]
) -> pd.DataFrame:
    long_frame = frame.melt(
        id_vars=group_columns,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return (
        long_frame.groupby([*group_columns, "metric"])["value"]
        .agg(count="count", mean="mean", std="std", median="median")
        .reset_index()
    )


def select_cases(records: list[dict], count: int) -> list[tuple[str, dict]]:
    if count <= 0 or not records:
        return []
    cliff_records = [record for record in records if record["cliff"] == 1]
    candidates = cliff_records or records
    candidates = sorted(candidates, key=lambda record: record["mean_abs_error"])
    indices = [0, len(candidates) // 2, len(candidates) - 1]
    labels = ["lowest_error", "median_error", "highest_error"]
    selected = []
    seen = set()
    for label, index in zip(labels, indices):
        record = candidates[index]
        if record["row_id"] not in seen:
            selected.append((label, record))
            seen.add(record["row_id"])
        if len(selected) >= count:
            break
    halogen_cases = []
    for record in records:
        mol = Chem.MolFromSmiles(record["smiles"])
        halogen_count = sum(
            atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()
        )
        if halogen_count >= 2 and record["row_id"] not in seen:
            contrast = float(np.ptp(record["mean_atom_delta"]))
            halogen_cases.append((contrast, record))
    if halogen_cases and len(selected) < count:
        _, record = max(halogen_cases, key=lambda item: item[0])
        selected.append(("multiple_halogen", record))
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate faithful chemical-relative-position and plain-gradient "
            "interpretability analyses from final adapted checkpoints."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/adapted")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("interpretability_outputs")
    )
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-cliff", type=int, default=10)
    parser.add_argument("--max-noncliff", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--render-cases-per-target", type=int, default=4)
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--skip-cases", action="store_true")
    parser.add_argument("--skip-statistical-figures", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-only", action="store_true", help="Check all requested checkpoint sources without generating explanations.")
    parser.add_argument(
        "--legacy-audit-manifest", type=Path,
        help="Required for historical checkpoints without embedded protocol fingerprints.",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if args.max_cliff < 0 or args.max_noncliff < 0:
        raise ValueError("Sampling limits must be non-negative")
    if args.max_cliff + args.max_noncliff == 0:
        raise ValueError("At least one sampling limit must be positive")
    if not args.skip_faithfulness and args.random_repeats < 1:
        raise ValueError("--random-repeats must be positive")
    device = torch.device(args.device)
    checkpoint_targets = sorted(
        path.name for path in args.checkpoint_dir.iterdir() if path.is_dir()
    )
    targets = checkpoint_targets if args.targets is None else args.targets
    unknown = sorted(set(targets) - set(checkpoint_targets))
    if unknown:
        raise ValueError(f"Targets without checkpoint directories: {unknown}")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate targets are not allowed")
    data = prepare_development_data(data_dir=args.data_dir)
    if sorted(checkpoint_targets) != data["target_names"]:
        raise RuntimeError("Checkpoint target directories differ from development data")
    audit = None
    audit_records = {}
    if args.legacy_audit_manifest is not None:
        audit = json.loads(args.legacy_audit_manifest.read_text())
        if audit.get("status") not in (
            "legacy_metadata_consistent_config_unverified",
            "legacy_metadata_consistent_local_development_fingerprint_mismatch",
        ):
            raise RuntimeError("Unexpected legacy-audit status")
        if audit.get("local_recomputed_development_data_sha256") != data["development_data_sha256"]:
            raise RuntimeError("Local development data changed since the legacy audit")
        if audit.get("target_names") != data["target_names"]:
            raise RuntimeError("Legacy audit target order mismatch")
        results_path = Path(audit["results_csv"])
        if sha256_file(results_path) != audit["results_csv_sha256"]:
            raise RuntimeError("Historical results table changed since the audit")
        for target, expected_hash in audit["input_csv_sha256"].items():
            if sha256_file(args.data_dir / f"{target}.csv") != expected_hash:
                raise RuntimeError(f"Input CSV changed since the audit: {target}")
        audit_records = {
            (item["target"], int(item["seed"])): item for item in audit["results"]
        }
        if len(audit_records) != len(audit["results"]):
            raise RuntimeError("Duplicate legacy-audit checkpoint identity")
    checkpoint_sources = []
    for target in targets:
        for seed in args.seeds:
            checkpoint_path = args.checkpoint_dir / target / f"chemwave_adapted_seed{seed}.pt"
            audit_record = audit_records.get((target, seed))
            if audit is not None and audit_record is None:
                raise RuntimeError(f"Checkpoint missing from legacy audit: {target} seed={seed}")
            _, checkpoint = load_model(
                checkpoint_path, target=target, seed=seed, device=device,
                expected_fingerprint=(
                    audit["development_data_sha256"] if audit is not None
                    else data["development_data_sha256"]
                ),
                expected_target_names=data["target_names"],
                legacy_sha256=(
                    audit_record["adapted_checkpoint_sha256"] if audit_record else None
                ),
            )
            checkpoint_sources.append({
                "target": target, "seed": seed,
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
                "protocol_sha256": checkpoint.get("protocol_sha256"),
            })
    if args.verify_only:
        print(json.dumps({
            "status": audit["status"] if audit is not None else "embedded_protocol_verified",
            "development_data_sha256": (
                audit["development_data_sha256"] if audit is not None
                else data["development_data_sha256"]
            ),
            "checkpoint_count": len(checkpoint_sources),
        }, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    molecule_rows, atom_rows, bond_rows, gate_rows, faith_rows = [], [], [], [], []
    explanation_store: dict[tuple[str, int], list[dict]] = {}
    row_store: dict[tuple[str, int], dict] = {}
    common_target_names = None
    source_by_key = {(item["target"], item["seed"]): item for item in checkpoint_sources}

    for target in targets:
        csv_path = args.data_dir / f"{target}.csv"
        frame = pd.read_csv(
            csv_path,
            usecols=["smiles", "y [pEC50/pKi]", "cliff_mol", "split"],
        )
        frame["row_id"] = frame.index
        frame["split"] = frame["split"].str.lower()
        frame = frame.loc[frame["split"] == "test"].dropna(
            subset=["smiles", "y [pEC50/pKi]", "cliff_mol"]
        )
        selected = stratified_sample(frame, args.max_cliff, args.max_noncliff)
        if selected.empty:
            raise RuntimeError(f"No sampled test molecules for {target}")
        target_gate_values = []

        for seed in args.seeds:
            checkpoint_path = (
                args.checkpoint_dir
                / target
                / f"chemwave_adapted_seed{seed}.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            if sha256_file(checkpoint_path) != source_by_key[(target, seed)]["sha256"]:
                raise RuntimeError(f"Checkpoint changed after provenance preflight: {checkpoint_path}")
            audit_record = audit_records.get((target, seed))
            if audit is not None and audit_record is None:
                raise RuntimeError(f"Checkpoint missing from legacy audit: {target} seed={seed}")
            model, checkpoint = load_model(
                checkpoint_path, target=target, seed=seed, device=device,
                expected_fingerprint=(
                    audit["development_data_sha256"] if audit is not None
                    else data["development_data_sha256"]
                ),
                expected_target_names=data["target_names"],
                legacy_sha256=(
                    audit_record["adapted_checkpoint_sha256"] if audit_record else None
                ),
            )
            if audit_record is not None and sha256_file(checkpoint_path) != audit_record["adapted_checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint differs from legacy audit: {checkpoint_path}")
            target_names = checkpoint["target_names"]
            if common_target_names is None:
                common_target_names = target_names
            elif target_names != common_target_names:
                raise RuntimeError("Checkpoint target order differs across runs")
            target_index = target_names.index(target)

            for _, row in selected.iterrows():
                graph = molecule_to_graph35(
                    row.smiles,
                    float(row["y [pEC50/pKi]"]),
                    target_index=target_index,
                )
                explanation = explain_molecule(model, graph)
                key = (target, int(row.row_id))
                explanation_store.setdefault(key, []).append(explanation)
                row_store[key] = {
                    "target": target,
                    "row_id": int(row.row_id),
                    "smiles": row.smiles,
                    "y": float(row["y [pEC50/pKi]"]),
                    "cliff": int(row.cliff_mol),
                }
                atom_delta = explanation["atom_prediction_delta"]
                bond_delta = np.asarray(
                    [bond["prediction_delta"] for bond in explanation["bonds"]]
                )
                mol = Chem.MolFromSmiles(row.smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES in {target}: {row.smiles}")
                heavy_atom_count = mol.GetNumHeavyAtoms()
                if heavy_atom_count < 1:
                    raise ValueError(
                        f"Molecule without heavy atoms in {target}: {row.smiles}"
                    )
                relative_total_abs = float(np.abs(atom_delta).sum())
                bond_total_abs = float(np.abs(bond_delta).sum())
                molecule_rows.append(
                    {
                        **row_store[key],
                        "seed": seed,
                        "prediction": explanation["prediction"],
                        "prediction_abs_error": abs(
                            explanation["prediction"] - float(row["y [pEC50/pKi]"])
                        ),
                        "heavy_atom_count": heavy_atom_count,
                        "relative_total_abs": relative_total_abs,
                        "relative_mean_abs_per_heavy_atom": (
                            relative_total_abs / heavy_atom_count
                        ),
                        "relative_max_abs": float(np.abs(atom_delta).max()),
                        "relative_top3_fraction": concentration(atom_delta),
                        "bond_total_abs": bond_total_abs,
                        "bond_mean_abs_per_heavy_atom": (
                            bond_total_abs / heavy_atom_count
                        ),
                        "bond_max_abs": float(np.abs(bond_delta).max())
                        if len(bond_delta)
                        else 0.0,
                        "bond_top3_fraction": concentration(bond_delta)
                        if len(bond_delta)
                        else 0.0,
                        "conductance_mean": float(
                            np.mean(
                                [
                                    bond["conductance_mean"]
                                    for bond in explanation["bonds"]
                                ]
                            )
                        )
                        if explanation["bonds"]
                        else 0.0,
                        "intrinsic_gradient_norm_mean": float(
                            np.mean(
                                [
                                    bond["gradient_norm_mean"]
                                    for bond in explanation["bonds"]
                                ]
                            )
                        )
                        if explanation["bonds"]
                        else 0.0,
                    }
                )
                for atom_index, atom in enumerate(mol.GetAtoms()):
                    atom_rows.append(
                        {
                            **row_store[key],
                            "seed": seed,
                            "atom_index": atom_index,
                            "element": atom.GetSymbol(),
                            "relative_prediction_delta": float(atom_delta[atom_index]),
                            "relative_injection_norm": float(
                                explanation["atom_relative_injection_norm"][atom_index]
                            ),
                            "frequency_gate_mean": float(
                                explanation["atom_frequency_gate_mean"][atom_index]
                            ),
                        }
                    )
                for bond_index, bond_record in enumerate(explanation["bonds"]):
                    begin, end = bond_record["atoms"]
                    chemical_bond = mol.GetBondBetweenAtoms(int(begin), int(end))
                    bond_rows.append(
                        {
                            **row_store[key],
                            "seed": seed,
                            "bond_index": bond_index,
                            "atom_i": begin,
                            "atom_j": end,
                            "bond_type": str(chemical_bond.GetBondType()),
                            "gradient_prediction_delta": bond_record[
                                "prediction_delta"
                            ],
                            "conductance_mean": bond_record["conductance_mean"],
                            "intrinsic_gradient_norm_mean": bond_record[
                                "gradient_norm_mean"
                            ],
                        }
                    )
                if not args.skip_faithfulness:
                    records = faithfulness_interventions(
                        model,
                        graph,
                        explanation,
                        random_repeats=args.random_repeats,
                        random_seed=seed * 100000 + int(row.row_id),
                    )
                    for record in records:
                        faith_rows.append({**row_store[key], "seed": seed, **record})

            gate = explanation_store[(target, int(selected.iloc[0].row_id))][-1][
                "effective_relative_gate"
            ]
            target_gate_values.append(gate)
            for dimension, value in enumerate(gate):
                radius_index, role_index = divmod(
                    dimension, len(CHEMICAL_ROLE_NAMES)
                )
                gate_rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "dimension": dimension,
                        "role": CHEMICAL_ROLE_NAMES[role_index],
                        "radius": RELATIVE_POSITION_RADII[radius_index],
                        "effective_gate": float(value),
                    }
                )
            del model

        if not args.skip_statistical_figures:
            plot_target_gate(
                np.mean(target_gate_values, axis=0),
                target,
                args.output_dir / "figures" / "target_gates" / f"{target}.png",
            )

    molecule_frame = pd.DataFrame(molecule_rows)
    atom_frame = pd.DataFrame(atom_rows)
    bond_frame = pd.DataFrame(bond_rows)
    gate_frame = pd.DataFrame(gate_rows)
    faith_frame = pd.DataFrame(faith_rows)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    molecule_frame.to_csv(tables_dir / "molecule_scores.csv", index=False)
    atom_frame.to_csv(tables_dir / "atom_scores.csv", index=False)
    bond_frame.to_csv(tables_dir / "bond_scores.csv", index=False)
    gate_frame.to_csv(tables_dir / "target_gate_scores.csv", index=False)
    if not faith_frame.empty:
        faith_frame.to_csv(tables_dir / "faithfulness.csv", index=False)

    metrics = [
        "prediction_abs_error",
        "heavy_atom_count",
        "relative_total_abs",
        "relative_mean_abs_per_heavy_atom",
        "relative_max_abs",
        "relative_top3_fraction",
        "bond_total_abs",
        "bond_mean_abs_per_heavy_atom",
        "bond_max_abs",
        "bond_top3_fraction",
        "conductance_mean",
        "intrinsic_gradient_norm_mean",
    ]
    group_summary = tidy_group_summary(molecule_frame, ["cliff"], metrics)
    group_summary.to_csv(tables_dir / "cliff_noncliff_summary.csv", index=False)
    target_group_summary = tidy_group_summary(
        molecule_frame, ["target", "cliff"], metrics
    )
    target_group_summary.to_csv(
        tables_dir / "cliff_noncliff_by_target.csv", index=False
    )

    stability_rows = []
    case_records_by_target: dict[str, list[dict]] = {}
    for key, explanations in explanation_store.items():
        target, row_id = key
        atom_values = [value["atom_prediction_delta"] for value in explanations]
        bond_values = [
            np.asarray([bond["prediction_delta"] for bond in value["bonds"]])
            for value in explanations
        ]
        stability_rows.append(
            {
                **row_store[key],
                "atom_spearman_mean": mean_pairwise(atom_values, spearman),
                "atom_top3_overlap_mean": mean_pairwise(atom_values, top_k_overlap),
                "bond_spearman_mean": mean_pairwise(bond_values, spearman),
                "bond_top3_overlap_mean": mean_pairwise(bond_values, top_k_overlap),
            }
        )
        mean_atom = np.mean(atom_values, axis=0)
        mean_bond = np.mean(bond_values, axis=0)
        mean_prediction = float(
            np.mean([value["prediction"] for value in explanations])
        )
        mean_bonds = copy.deepcopy(explanations[0]["bonds"])
        for bond_index, bond in enumerate(mean_bonds):
            bond["prediction_delta"] = float(mean_bond[bond_index])
            bond["conductance_mean"] = float(
                np.mean(
                    [value["bonds"][bond_index]["conductance_mean"] for value in explanations]
                )
            )
            bond["gradient_norm_mean"] = float(
                np.mean(
                    [value["bonds"][bond_index]["gradient_norm_mean"] for value in explanations]
                )
            )
        case_records_by_target.setdefault(target, []).append(
            {
                **row_store[key],
                "mean_prediction": mean_prediction,
                "mean_abs_error": abs(mean_prediction - row_store[key]["y"]),
                "mean_atom_delta": mean_atom,
                "mean_bonds": mean_bonds,
            }
        )
    stability_frame = pd.DataFrame(stability_rows)
    stability_frame.to_csv(tables_dir / "seed_stability.csv", index=False)

    if not args.skip_cases:
        for target, records in case_records_by_target.items():
            for label, record in select_cases(records, args.render_cases_per_target):
                draw_molecule_explanation(
                    record["smiles"],
                    record["mean_atom_delta"],
                    record["mean_bonds"],
                    args.output_dir
                    / "figures"
                    / "molecule_cases"
                    / target
                    / f"row_{record['row_id']}_{label}.png",
                    legend=(
                        f"{target} | {label} | cliff={record['cliff']} | "
                        f"y={record['y']:.3f} pred={record['mean_prediction']:.3f}\n"
                        "atoms: relative intervention; bonds: bond-difference intervention; "
                        "red raises / blue lowers"
                    ),
                )

    figures_dir = args.output_dir / "figures"
    if not args.skip_statistical_figures:
        plot_all_target_gates(gate_frame, figures_dir / "all_target_gates.png")
        plot_all_target_gates(gate_frame, figures_dir / "all_target_gates.pdf")
        plot_cliff_noncliff(molecule_frame, figures_dir / "cliff_noncliff.png")
        if not faith_frame.empty:
            plot_faithfulness(faith_frame, figures_dir / "faithfulness.png")
        if not stability_frame.empty and len(args.seeds) > 1:
            plot_stability(stability_frame, figures_dir / "seed_stability.png")

    manifest = {
        "provenance_status": (
            audit["status"] if audit is not None
            else "embedded_protocol_verified"
        ),
        "development_data_sha256": (
            audit["development_data_sha256"] if audit is not None
            else data["development_data_sha256"]
        ),
        "local_recomputed_development_data_sha256": data["development_data_sha256"],
        "input_csv_sha256": {
            target: sha256_file(args.data_dir / f"{target}.csv")
            for target in data["target_names"]
        },
        "legacy_audit_manifest": (
            str(args.legacy_audit_manifest.resolve()) if audit is not None else None
        ),
        "legacy_audit_manifest_sha256": (
            sha256_file(args.legacy_audit_manifest) if audit is not None else None
        ),
        "checkpoints": checkpoint_sources,
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "targets": targets,
        "seeds": args.seeds,
        "max_cliff_per_target": args.max_cliff,
        "max_noncliff_per_target": args.max_noncliff,
        "random_repeats": args.random_repeats,
        "statistical_figures_skipped": args.skip_statistical_figures,
        "molecule_seed_records": len(molecule_frame),
        "model_input_uses_cliff_label": False,
        "interpretation": {
            "atom_score": "prediction change after zeroing one atom's relative-position channel",
            "bond_score": "prediction change after zeroing one chemical bond in the gradient branch only",
            "conductance": "intrinsic transmission coefficient, not prediction importance",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
