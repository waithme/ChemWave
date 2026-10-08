from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from chemwave_multitask import VARIANT_CHOICES


DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def run(command: list[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated ChemWave ablation: five-seed shared pretraining, "
            "then five-seed adaptation and testing on all 30 datasets."
        )
    )
    parser.add_argument("--variant", choices=VARIANT_CHOICES, required=True,
                        help="Final Full: a5_full; affine comparator: a3_affine_transport. Legacy aliases remain accepted.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiment2"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("final_based_ablation_runs")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--expected-target-count", type=int, default=30)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    variant_root = output_root / args.variant
    pretrain_dir = variant_root / "checkpoints" / "pretrain"
    adapted_dir = variant_root / "checkpoints" / "adapted"
    results_dir = variant_root / "results"
    pretrain_json = results_dir / "pretrain.json"
    results_csv = results_dir / "all_results.csv"
    seed_args = [str(seed) for seed in DEFAULT_SEEDS]

    common = [
        "--variant",
        args.variant,
        "--data-dir",
        str(data_dir),
        "--pretrain-dir",
        str(pretrain_dir),
        "--seeds",
        *seed_args,
        "--hidden-dim",
        str(args.hidden_dim),
        "--device",
        args.device,
    ]
    run(
        [
            sys.executable,
            str(project_dir / "run_chemwave_pretrain.py"),
            *common,
            "--pretrain-epochs",
            str(args.pretrain_epochs),
            "--output",
            str(pretrain_json),
        ],
        project_dir,
    )
    run(
        [
            sys.executable,
            str(project_dir / "run_chemwave_finetune_test.py"),
            *common,
            "--adapted-dir",
            str(adapted_dir),
            "--results-csv",
            str(results_csv),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--expected-target-count",
            str(args.expected_target_count),
        ],
        project_dir,
    )
    print(f"completed variant={args.variant} results={results_csv}", flush=True)


if __name__ == "__main__":
    main()
