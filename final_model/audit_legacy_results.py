"""Read-only verification of historical Full results; writes only a sidecar JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd
import torch

from chemwave_provenance import VARIANT, sha256_file
from chemwave_multitask import TargetConditionedChemWave
from chemwave_training import prepare_development_data
from run_chemwave_finetune_test import load_test_graphs, test_metrics


def _same_number(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)


def audit(
    *, results_csv: Path, pretrain_dir: Path, adapted_dir: Path,
    data_dir: Path, expected_count: int, pretrain_summary: Path | None = None,
    recompute_test_metrics: bool = False,
) -> dict:
    with results_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} result rows, found {len(rows)}")
    if not rows:
        raise RuntimeError("Empty results table")
    keys = [(row["target"], int(row["seed"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate target/seed result key")
    data = prepare_development_data(data_dir=data_dir)
    local_fingerprint = data["development_data_sha256"]
    fingerprints = {row["development_data_sha256"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("Result table mixes development fingerprints")
    fingerprint = fingerprints.pop()
    target_names = data["target_names"]
    seeds = sorted({int(row["seed"]) for row in rows})
    expected_keys = {(target, seed) for target in target_names for seed in seeds}
    if set(keys) != expected_keys:
        raise RuntimeError("Result target/seed grid is incomplete or contains unknown targets")
    pretrain_records = {}
    for seed in seeds:
        path = pretrain_dir / f"chemwave_shared_pretrain_seed{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for field, expected in {
            "stage": "shared_pretrain", "variant": VARIANT,
            "development_data_sha256": fingerprint,
            "target_names": target_names, "seed": seed,
        }.items():
            if checkpoint.get(field) != expected:
                raise RuntimeError(f"Pretrain {path} mismatched {field}")
        if "model_state" not in checkpoint:
            raise RuntimeError(f"No model_state in {path}")
        if tuple(checkpoint["model_state"]["output_weight"].shape) != (len(target_names), 300):
            raise RuntimeError(f"Unexpected pretrain model width/target count: {path}")
        pretrain_records[seed] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "best_epoch": int(checkpoint["best_epoch"]),
            "validation_rmse": float(checkpoint["validation_rmse"]),
            "has_protocol_fingerprint": "protocol_sha256" in checkpoint,
        }

    pretrain_summary = pretrain_summary or results_csv.with_name("pretrain.json")
    summary = json.loads(pretrain_summary.read_text())
    for field, expected in {
        "stage": "shared_pretrain", "variant": VARIANT,
        "development_data_sha256": fingerprint, "target_names": target_names,
    }.items():
        if summary.get(field) != expected:
            raise RuntimeError(f"Pretrain summary mismatched {field}: {pretrain_summary}")
    summary_runs = {int(item["seed"]): item for item in summary["runs"]}
    if len(summary_runs) != len(summary["runs"]) or set(summary_runs) != set(seeds):
        raise RuntimeError("Pretrain summary seed grid does not match results")
    for seed, item in summary_runs.items():
        if int(item["best_epoch"]) != pretrain_records[seed]["best_epoch"]:
            raise RuntimeError(f"Pretrain summary epoch mismatch: seed={seed}")
        if not _same_number(item["validation_rmse"], pretrain_records[seed]["validation_rmse"]):
            raise RuntimeError(f"Pretrain summary validation mismatch: seed={seed}")
        if Path(item["checkpoint"]).resolve() != Path(pretrain_records[seed]["path"]):
            raise RuntimeError(f"Pretrain summary checkpoint path mismatch: seed={seed}")

    data_files = {
        target: sha256_file(data_dir / f"{target}.csv") for target in target_names
    }
    counts = {}
    for target in target_names:
        frame = pd.read_csv(
            data_dir / f"{target}.csv",
            usecols=["smiles", "y [pEC50/pKi]", "cliff_mol", "split"],
        )
        frame = frame.loc[frame["split"].str.lower() == "test"].dropna(
            subset=["smiles", "y [pEC50/pKi]", "cliff_mol"]
        )
        counts[target] = (len(frame), int(frame["cliff_mol"].astype(int).sum()))

    result_records = []
    max_metric_absolute_difference = 0.0
    cached_target = None
    cached_test_graphs = None
    for row in rows:
        target, seed = row["target"], int(row["seed"])
        path = adapted_dir / target / f"chemwave_adapted_seed{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for field, expected in {
            "stage": "adapted", "variant": VARIANT,
            "development_data_sha256": fingerprint,
            "target_names": target_names, "target_name": target, "seed": seed,
        }.items():
            if checkpoint.get(field) != expected:
                raise RuntimeError(f"Adapted {path} mismatched {field}")
        if row["variant"] != VARIANT or row["development_data_sha256"] != fingerprint:
            raise RuntimeError(f"Result metadata mismatch: {target} seed={seed}")
        if Path(row["pretrain_checkpoint"]).resolve() != Path(pretrain_records[seed]["path"]):
            raise RuntimeError(f"Result points at another pretrain checkpoint: {target} seed={seed}")
        if Path(row["adapted_checkpoint"]).resolve() != path.resolve():
            raise RuntimeError(f"Result points at another adapted checkpoint: {target} seed={seed}")
        if Path(checkpoint["shared_pretrain_checkpoint"]).name != Path(pretrain_records[seed]["path"]).name:
            raise RuntimeError(f"Adapted checkpoint names another pretrain seed: {target} seed={seed}")
        if int(row["pretrain_best_epoch"]) != pretrain_records[seed]["best_epoch"]:
            raise RuntimeError(f"Pretrain epoch mismatch: {target} seed={seed}")
        if not _same_number(row["pretrain_val_rmse"], pretrain_records[seed]["validation_rmse"]):
            raise RuntimeError(f"Pretrain validation mismatch: {target} seed={seed}")
        if int(row["finetune_best_epoch"]) != int(checkpoint["best_epoch"]):
            raise RuntimeError(f"Adaptation epoch mismatch: {target} seed={seed}")
        if not _same_number(row["finetune_val_rmse"], checkpoint["validation_rmse"]):
            raise RuntimeError(f"Adaptation validation mismatch: {target} seed={seed}")
        if (int(row["test_count"]), int(row["cliff_test_count"])) != counts[target]:
            raise RuntimeError(f"Test counts mismatch: {target} seed={seed}")
        if not all(math.isfinite(float(row[field])) for field in ("test_rmse", "test_cliff_rmse")):
            raise RuntimeError(f"Non-finite test metric: {target} seed={seed}")
        if "model_state" not in checkpoint:
            raise RuntimeError(f"No model_state in {path}")
        if tuple(checkpoint["model_state"]["output_weight"].shape) != (len(target_names), 300):
            raise RuntimeError(f"Unexpected adapted model width/target count: {path}")
        if recompute_test_metrics:
            if cached_target != target:
                cached_test_graphs = load_test_graphs(
                    data_dir, target, target_names.index(target)
                )
                cached_target = target
            model = TargetConditionedChemWave(
                num_targets=len(target_names), hidden_dim=300
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            found = test_metrics(model, cached_test_graphs, torch.device("cpu"))
            for field in ("test_count", "cliff_test_count"):
                if int(row[field]) != found[field]:
                    raise RuntimeError(f"Recomputed {field} differs: {target} seed={seed}")
            for field in ("test_rmse", "test_cliff_rmse"):
                difference = abs(float(row[field]) - found[field])
                max_metric_absolute_difference = max(max_metric_absolute_difference, difference)
                if difference > 1e-6:
                    raise RuntimeError(
                        f"Recomputed {field} differs by {difference}: {target} seed={seed}"
                    )
            del model
        result_records.append({
            "target": target, "seed": seed,
            "pretrain_checkpoint_sha256": pretrain_records[seed]["sha256"],
            "adapted_checkpoint": str(path.resolve()),
            "adapted_checkpoint_sha256": sha256_file(path),
            "has_protocol_fingerprint": "protocol_sha256" in checkpoint,
        })

    fingerprint_reproduced = local_fingerprint == fingerprint
    return {
        "audit_version": 1,
        "status": (
            "legacy_metadata_consistent_config_unverified" if fingerprint_reproduced
            else "legacy_metadata_consistent_local_development_fingerprint_mismatch"
        ),
        "limits": [
            "Historical checkpoints do not contain a full architecture/training configuration fingerprint.",
            *([] if recompute_test_metrics else [
                "Matching checkpoint metadata and validation summaries do not prove that test metrics were recomputed from these exact weights."
            ]),
            "The embedded pretrain path may be a historical server location; the result table's local path and seed are checked instead.",
            *([] if fingerprint_reproduced else [
                "Rebuilding development data from the current local CSVs and installed RDKit does not reproduce the historical checkpoint fingerprint; the original training input snapshot or environment has not been independently recovered."
            ]),
        ],
        "variant": VARIANT,
        "development_data_sha256": fingerprint,
        "local_recomputed_development_data_sha256": local_fingerprint,
        "development_fingerprint_reproduced": fingerprint_reproduced,
        "target_names": target_names,
        "seeds": seeds,
        "results_csv": str(results_csv.resolve()),
        "results_csv_sha256": sha256_file(results_csv),
        "pretrain_summary": str(pretrain_summary.resolve()),
        "pretrain_summary_sha256": sha256_file(pretrain_summary),
        "input_csv_sha256": data_files,
        "result_row_count": len(rows),
        "test_metrics_recomputed_from_exact_checkpoints": recompute_test_metrics,
        "maximum_test_metric_absolute_difference": (
            max_metric_absolute_difference if recompute_test_metrics else None
        ),
        "pretrain": pretrain_records,
        "results": result_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=150)
    parser.add_argument("--pretrain-summary", type=Path)
    parser.add_argument(
        "--recompute-test-metrics", action="store_true",
        help="Read-only CPU inference for all result rows; no training or checkpoint writes.",
    )
    args = parser.parse_args()
    report = audit(
        results_csv=args.results_csv, pretrain_dir=args.pretrain_dir,
        adapted_dir=args.adapted_dir, data_dir=args.data_dir,
        expected_count=args.expected_count, pretrain_summary=args.pretrain_summary,
        recompute_test_metrics=args.recompute_test_metrics,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "rows": report["result_row_count"],
        "pretrain_checkpoints": len(report["pretrain"]),
        "adapted_checkpoints": len(report["results"]),
        "test_metrics_recomputed": report["test_metrics_recomputed_from_exact_checkpoints"],
        "maximum_metric_absolute_difference": report["maximum_test_metric_absolute_difference"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
