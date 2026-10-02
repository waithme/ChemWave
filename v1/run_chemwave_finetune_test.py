from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from chemwave_multitask import TargetConditionedChemWave
from chemwave_features import molecule_to_graph35
from chemwave_training import (
    MODEL_LOCK,
    ensure_target_adaptation,
    load_checkpoint,
    load_model_lock,
    prepare_development_data,
    select_target_data,
    validate_checkpoint,
)


EVALUATION_REGISTRY = Path("CHEMWAVE_EVALUATION_REGISTRY.json")
CSV_FIELDS = [
    "model_id",
    "model_lock_sha256",
    "development_data_sha256",
    "target",
    "seed",
    "evaluation_status",
    "pretrain_checkpoint",
    "adapted_checkpoint",
    "pretrain_best_epoch",
    "pretrain_val_rmse",
    "finetune_best_epoch",
    "finetune_val_rmse",
    "test_count",
    "cliff_test_count",
    "test_rmse",
    "test_cliff_rmse",
    "finished_at_utc",
]


def result_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["model_id"]),
        str(row["development_data_sha256"]),
        str(row["target"]),
        str(row["seed"]),
    )


def csv_contains(path: Path, key: tuple[str, str, str, str]) -> bool:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open(newline="") as result_file:
            reader = csv.DictReader(result_file)
            if reader.fieldnames != CSV_FIELDS:
                raise RuntimeError(
                    f"Unexpected CSV schema in {path}: {reader.fieldnames}"
                )
            return any(result_key(row) == key for row in reader)


def append_csv_once(path: Path, row: dict) -> bool:
    key = result_key(row)
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists() and path.stat().st_size > 0:
            with path.open(newline="") as result_file:
                reader = csv.DictReader(result_file)
                if reader.fieldnames != CSV_FIELDS:
                    raise RuntimeError(
                        f"Unexpected CSV schema in {path}: {reader.fieldnames}"
                    )
                if any(result_key(existing) == key for existing in reader):
                    return False
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow({field: row[field] for field in CSV_FIELDS})
            result_file.flush()
            os.fsync(result_file.fileno())
        return True


def load_test_graphs(data_dir: Path, target_name: str, target_index: int):
    path = data_dir / f"{target_name}.csv"
    frame = pd.read_csv(
        path,
        usecols=["smiles", "y [pEC50/pKi]", "cliff_mol", "split"],
    )
    frame["split"] = frame["split"].str.lower()
    frame = frame.loc[frame["split"] == "test"].dropna(
        subset=["smiles", "y [pEC50/pKi]", "cliff_mol"]
    )
    graphs = [
        molecule_to_graph35(
            row.smiles,
            row["y [pEC50/pKi]"],
            cliff_label=int(row.cliff_mol),
            target_index=target_index,
        )
        for _, row in frame.iterrows()
    ]
    if not graphs:
        raise RuntimeError(f"Empty official test partition for {target_name}")
    if any("cliff_label" not in graph.keys() for graph in graphs):
        raise AssertionError("Test graph is missing cliff_label")
    return graphs


@torch.no_grad()
def test_metrics(model, graphs, device: torch.device) -> dict:
    model.eval()
    predictions, targets, cliffs = [], [], []
    loader = DataLoader(graphs, batch_size=128, shuffle=False, num_workers=0)
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).cpu())
        targets.append(batch.y.view(-1).cpu())
        cliffs.append(batch.cliff_label.view(-1).bool().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    cliff = torch.cat(cliffs)
    if not torch.any(cliff):
        raise RuntimeError("Official test partition contains no cliff molecules")
    return {
        "test_count": int(target.numel()),
        "cliff_test_count": int(cliff.sum().item()),
        "test_rmse": torch.sqrt(torch.mean((prediction - target).square())).item(),
        "test_cliff_rmse": torch.sqrt(
            torch.mean((prediction[cliff] - target[cliff]).square())
        ).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load shared pretraining, then fine-tune and test every requested "
            "dataset. Results are process-safely appended to one CSV."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset; default is every CHEMBL*.csv under --data-dir.",
    )
    parser.add_argument("--expected-target-count", type=int, default=30)
    parser.add_argument("--model-lock", type=Path, default=MODEL_LOCK)
    parser.add_argument(
        "--evaluation-registry", type=Path, default=EVALUATION_REGISTRY
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    lock, lock_sha256 = load_model_lock(args.model_lock)
    args.hidden_dim = int(lock["architecture"]["hidden_dim"])
    args.finetune_epochs = int(lock["training_protocol"]["maximum_epochs"])
    device = torch.device(args.device)
    data = prepare_development_data(data_dir=args.data_dir)
    if len(data["target_names"]) != args.expected_target_count:
        raise RuntimeError(
            f"Expected {args.expected_target_count} datasets, found "
            f"{len(data['target_names'])} under {args.data_dir}"
        )
    targets = data["target_names"] if args.targets is None else args.targets
    unknown = sorted(set(targets) - set(data["target_names"]))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate targets are not allowed")

    registry = json.loads(args.evaluation_registry.read_text())
    preexisting_contaminated = set(registry["development_contaminated_targets"])
    print(
        f"device={device} datasets={len(data['target_names'])} "
        f"selected_targets={len(targets)} seeds={args.seeds}",
        flush=True,
    )

    appended = skipped = 0
    for seed in args.seeds:
        pretrain_path = (
            args.pretrain_dir / f"chemwave_shared_pretrain_seed{seed}.pt"
        )
        if not pretrain_path.exists():
            raise FileNotFoundError(
                f"Missing {pretrain_path}. Run run_chemwave_pretrain.py first."
            )
        pretrain = load_checkpoint(pretrain_path, device)
        validate_checkpoint(
            pretrain,
            stage="shared_pretrain",
            seed=seed,
            data=data,
            lock=lock,
            lock_sha256=lock_sha256,
        )
        for target_name in targets:
            key = (
                lock["model_id"],
                data["development_data_sha256"],
                target_name,
                str(seed),
            )
            if csv_contains(args.results_csv, key):
                skipped += 1
                print(f"skipping completed result: {target_name} seed={seed}", flush=True)
                continue

            target_data = select_target_data(data, target_name)
            args.target_name = target_name
            adapted, adaptation_source, adapted_path = ensure_target_adaptation(
                pretrain, target_data, seed, args, device, lock, lock_sha256
            )
            model = TargetConditionedChemWave(
                num_targets=len(data["target_names"]), hidden_dim=args.hidden_dim
            ).to(device)
            model.load_state_dict(adapted["model_state"])
            test_graphs = load_test_graphs(
                args.data_dir, target_name, target_data["target_index"]
            )
            metrics = test_metrics(model, test_graphs, device)
            row = {
                "model_id": lock["model_id"],
                "model_lock_sha256": lock_sha256,
                "development_data_sha256": data["development_data_sha256"],
                "target": target_name,
                "seed": seed,
                "evaluation_status": (
                    "exploratory_development_contaminated"
                    if target_name in preexisting_contaminated
                    else "locked_final_batch_test"
                ),
                "pretrain_checkpoint": str(pretrain_path),
                "adapted_checkpoint": str(adapted_path),
                "pretrain_best_epoch": int(pretrain["best_epoch"]),
                "pretrain_val_rmse": float(pretrain["validation_rmse"]),
                "finetune_best_epoch": int(adapted["best_epoch"]),
                "finetune_val_rmse": float(adapted["validation_rmse"]),
                **metrics,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if append_csv_once(args.results_csv, row):
                appended += 1
                print(json.dumps({**row, "adaptation_source": adaptation_source}), flush=True)
            else:
                skipped += 1
                print(
                    f"result was appended by another process: {target_name} seed={seed}",
                    flush=True,
                )
            del model, test_graphs

    print(
        json.dumps(
            {
                "results_csv": str(args.results_csv),
                "appended": appended,
                "skipped_existing": skipped,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
