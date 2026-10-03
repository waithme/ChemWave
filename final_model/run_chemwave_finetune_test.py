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
from chemwave_provenance import MODEL_OPTIONS, VARIANT, protocol, sha256_file
from chemwave_training import (
    ensure_target_adaptation,
    load_checkpoint,
    prepare_development_data,
    select_target_data,
    validate_checkpoint,
)


CSV_FIELDS = [
    "variant",
    "development_data_sha256",
    "target",
    "seed",
    "pretrain_protocol_sha256",
    "adapted_protocol_sha256",
    "pretrain_checkpoint_sha256",
    "adapted_checkpoint_sha256",
    "test_input_sha256",
    "evaluation_source_sha256",
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


def result_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row["development_data_sha256"]),
        str(row["target"]),
        str(row["seed"]),
    )


ORIGIN_FIELDS = (
    "variant", "pretrain_protocol_sha256", "adapted_protocol_sha256",
    "pretrain_checkpoint_sha256", "adapted_checkpoint_sha256",
    "test_input_sha256", "evaluation_source_sha256",
)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as result_file:
        reader = csv.DictReader(result_file)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"Unexpected CSV schema in {path}: {reader.fieldnames}. "
                "Preserve the historical table and choose a fresh results path."
            )
        return list(reader)


def _matches_existing(existing: dict, row: dict) -> bool:
    if any(str(existing[field]) != str(row[field]) for field in ORIGIN_FIELDS):
        raise RuntimeError(
            "Result already exists for this data/target/seed but its protocol, "
            "checkpoint, test input, or evaluation code differs; refusing to "
            "silently skip or mix results. Use a separate output directory."
        )
    return True


def csv_contains(path: Path, row: dict) -> bool:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        matches = [existing for existing in _read_rows(path) if result_key(existing) == result_key(row)]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate result key in {path}: {result_key(row)}")
        return bool(matches and _matches_existing(matches[0], row))


def append_csv_once(path: Path, row: dict) -> bool:
    key = result_key(row)
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        matches = [existing for existing in _read_rows(path) if result_key(existing) == key]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate result key in {path}: {key}")
        if matches:
            _matches_existing(matches[0], row)
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
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
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
    # Fail before any adaptation if a historical table is passed to the v2 runner.
    _read_rows(args.results_csv)

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
        pretrain_protocol = protocol(
            stage="shared_pretrain", data=data, seed=seed,
            hidden_dim=args.hidden_dim, max_epochs=args.pretrain_epochs,
        )
        validate_checkpoint(
            pretrain,
            stage="shared_pretrain",
            seed=seed,
            data=data,
            expected_protocol=pretrain_protocol,
        )
        pretrain_hash = sha256_file(pretrain_path)
        for target_name in targets:
            target_data = select_target_data(data, target_name)
            args.target_name = target_name
            expected_adapted_path = (
                args.adapted_dir / target_name / f"chemwave_adapted_seed{seed}.pt"
            )
            if not expected_adapted_path.exists():
                existing = [
                    row for row in _read_rows(args.results_csv)
                    if result_key(row) == (data["development_data_sha256"], target_name, str(seed))
                ]
                if existing:
                    raise RuntimeError(
                        f"Result exists but its adapted checkpoint is missing: {expected_adapted_path}"
                    )
            adapted, adaptation_source, adapted_path = ensure_target_adaptation(
                pretrain, target_data, seed, args, device
            )
            origin = {
                "variant": VARIANT,
                "development_data_sha256": data["development_data_sha256"],
                "target": target_name,
                "seed": seed,
                "pretrain_protocol_sha256": pretrain["protocol_sha256"],
                "adapted_protocol_sha256": adapted["protocol_sha256"],
                "pretrain_checkpoint_sha256": pretrain_hash,
                "adapted_checkpoint_sha256": sha256_file(adapted_path),
                "test_input_sha256": sha256_file(args.data_dir / f"{target_name}.csv"),
                "evaluation_source_sha256": sha256_file(Path(__file__)),
            }
            if csv_contains(args.results_csv, origin):
                skipped += 1
                print(f"skipping provenance-matched result: {target_name} seed={seed}", flush=True)
                continue
            model = TargetConditionedChemWave(
                num_targets=len(data["target_names"]),
                hidden_dim=args.hidden_dim,
                **MODEL_OPTIONS,
            ).to(device)
            model.load_state_dict(adapted["model_state"])
            test_graphs = load_test_graphs(
                args.data_dir, target_name, target_data["target_index"]
            )
            metrics = test_metrics(model, test_graphs, device)
            row = {
                **origin,
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
