from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from chemwave_multitask import TargetConditionedChemWave
from chemwave_training import (
    ensure_shared_pretrain,
    ensure_target_adaptation,
    prepare_development_data,
    select_target_data,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_report(num_targets: int, hidden_dim: int) -> dict:
    model = TargetConditionedChemWave(
        num_targets=num_targets, hidden_dim=hidden_dim
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    added = (
        model.relative_position_projection.weight.numel()
        + model.target_relative_position.weight.numel()
        + sum(
            block.edge_conductance.weight.numel()
            + block.target_bond_conductance.weight.numel()
            + block.target_gradient_gain.weight.numel()
            for block in model.blocks
        )
    )
    effective_adaptation = (
        sum(
            block.target_frequency.embedding_dim
            + block.target_bond_conductance.embedding_dim
            + block.target_gradient_gain.embedding_dim
            for block in model.blocks
        )
        + model.target_relative_position.embedding_dim
        + model.output_weight.shape[1]
        + 1
    )
    return {
        "full_total_parameters": int(total),
        "base_total_parameters": int(total - added),
        "full_added_parameters": int(added),
        "full_added_percent_of_base": float(100.0 * added / (total - added)),
        "effective_adaptation_parameters_per_target": int(effective_adaptation),
        "effective_adaptation_percent_of_full": float(
            100.0 * effective_adaptation / total
        ),
    }


def hardware_report(device: torch.device) -> dict:
    payload = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return payload


def inferred_executed_epochs(best_epoch: int, maximum: int, patience: int = 15) -> int:
    """Infer loop epochs from the fixed early-stopping rule in fit_stage."""
    return min(maximum, best_epoch + patience)


def write_markdown(payload: dict, path: Path) -> None:
    p = payload["parameters"]
    timing = payload.get("timing")
    lines = [
        "# ChemWave efficiency report",
        "",
        "This report excludes molecular graph preprocessing and measures one model seed.",
        "CUDA is synchronized immediately before and after every timed region.",
        "",
        "| Model | Total parameters | Added vs Base | Effective adaptation/target | Joint time | Adaptation time/target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if timing is None:
        joint = adapt = "not measured (`--parameter-only`)"
    else:
        joint = f"{timing['joint_training_seconds']:.1f} s"
        adapt = f"{timing['adaptation_seconds_mean']:.1f} s"
    lines.append(
        f"| ChemWave | {p['full_total_parameters']:,} | "
        f"{p['full_added_parameters']:,} ({p['full_added_percent_of_base']:.2f}%) | "
        f"{p['effective_adaptation_parameters_per_target']:,} "
        f"({p['effective_adaptation_percent_of_full']:.3f}%) | {joint} | {adapt} |"
    )
    if timing is not None:
        lines += [
            "",
            f"Joint best epoch: {timing['joint_best_epoch']}; inferred executed epochs: {timing['joint_executed_epochs']}.",
            f"All-target adaptation time: {timing['adaptation_seconds_total']:.1f} s for {timing['target_count']} targets.",
            f"Mean/median adaptation time: {timing['adaptation_seconds_mean']:.1f}/{timing['adaptation_seconds_median']:.1f} s per target.",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a minimal ChemWave efficiency profile: parameters, one-seed "
            "joint training time, and mean target-adaptation time."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--expected-target-count", type=int, default=30)
    parser.add_argument(
        "--parameter-only",
        action="store_true",
        help="Write the parameter report without loading data or training.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "efficiency.json"
    output_md = args.output_dir / "EFFICIENCY_REPORT.md"
    params = parameter_report(args.expected_target_count, args.hidden_dim)
    device = torch.device(args.device)
    payload = {
        "protocol": "minimal_efficiency_v1",
        "seed": args.seed,
        "parameters": params,
        "hardware": hardware_report(device),
        "timing_scope": {
            "graph_preprocessing_included": False,
            "checkpoint_write_included": True,
            "cuda_synchronized": device.type == "cuda",
            "model_seeds_measured": 1,
        },
    }
    if args.parameter_only:
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        write_markdown(payload, output_md)
        print(json.dumps(payload, indent=2), flush=True)
        return

    existing_checkpoints = sorted(args.output_dir.rglob("*.pt"))
    if existing_checkpoints:
        raise RuntimeError(
            "Efficiency timing requires a fresh output directory; existing "
            f"checkpoint found: {existing_checkpoints[0]}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    preprocess_start = time.perf_counter()
    data = prepare_development_data(data_dir=args.data_dir.resolve())
    preprocess_seconds = time.perf_counter() - preprocess_start
    if len(data["target_names"]) != args.expected_target_count:
        raise RuntimeError(
            f"Expected {args.expected_target_count} targets, found "
            f"{len(data['target_names'])}"
        )

    # Initialize the CUDA context before timing model training.
    if device.type == "cuda":
        torch.ones(1, device=device).add_(1.0)
        synchronize(device)

    pretrain_dir = args.output_dir / "checkpoints/pretrain"
    adapted_dir = args.output_dir / "checkpoints/adapted"
    training_args = SimpleNamespace(
        pretrain_dir=pretrain_dir,
        adapted_dir=adapted_dir,
        hidden_dim=args.hidden_dim,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        target_name=None,
    )

    synchronize(device)
    start = time.perf_counter()
    pretrain, source, pretrain_path = ensure_shared_pretrain(
        data, args.seed, training_args, device
    )
    synchronize(device)
    joint_seconds = time.perf_counter() - start
    if source != "trained":
        raise RuntimeError("Joint checkpoint was reused; timing is invalid")

    adaptation_csv = args.output_dir / "adaptation_times.csv"
    adaptation_rows = []
    for target_name in data["target_names"]:
        target_data = select_target_data(data, target_name)
        training_args.target_name = target_name
        synchronize(device)
        start = time.perf_counter()
        adapted, adaptation_source, adapted_path = ensure_target_adaptation(
            pretrain, target_data, args.seed, training_args, device
        )
        synchronize(device)
        seconds = time.perf_counter() - start
        if adaptation_source != "trained":
            raise RuntimeError(f"Adaptation checkpoint was reused for {target_name}")
        row = {
            "target": target_name,
            "seed": args.seed,
            "seconds": seconds,
            "best_epoch": int(adapted["best_epoch"]),
            "inferred_executed_epochs": inferred_executed_epochs(
                int(adapted["best_epoch"]), args.finetune_epochs
            ),
            "validation_rmse": float(adapted["validation_rmse"]),
            "checkpoint": str(adapted_path),
        }
        adaptation_rows.append(row)
        print(json.dumps(row), flush=True)

    with adaptation_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(adaptation_rows[0]))
        writer.writeheader()
        writer.writerows(adaptation_rows)

    times = np.asarray([row["seconds"] for row in adaptation_rows])
    payload.update(
        {
            "development_data_sha256": data["development_data_sha256"],
            "preprocessing_seconds_excluded": preprocess_seconds,
            "timing": {
                "joint_training_seconds": joint_seconds,
                "joint_best_epoch": int(pretrain["best_epoch"]),
                "joint_executed_epochs": inferred_executed_epochs(
                    int(pretrain["best_epoch"]), args.pretrain_epochs
                ),
                "joint_checkpoint": str(pretrain_path),
                "target_count": len(adaptation_rows),
                "adaptation_seconds_total": float(times.sum()),
                "adaptation_seconds_mean": float(times.mean()),
                "adaptation_seconds_median": float(np.median(times)),
                "adaptation_seconds_min": float(times.min()),
                "adaptation_seconds_max": float(times.max()),
            },
        }
    )
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload, output_md)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
