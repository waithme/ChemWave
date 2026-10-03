from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def run(command: list[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run five-seed shared pretraining, followed by five-seed target "
            "adaptation and official testing on all 30 datasets."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("final_runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--expected-target-count", type=int, default=30)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    pretrain_dir = output_dir / "checkpoints" / "pretrain"
    adapted_dir = output_dir / "checkpoints" / "adapted"
    results_dir = output_dir / "results"
    seeds = [str(seed) for seed in DEFAULT_SEEDS]

    common = [
        "--data-dir",
        str(data_dir),
        "--pretrain-dir",
        str(pretrain_dir),
        "--seeds",
        *seeds,
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
            str(results_dir / "pretrain.json"),
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
            str(results_dir / "all_results.csv"),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--pretrain-epochs",
            str(args.pretrain_epochs),
            "--expected-target-count",
            str(args.expected_target_count),
        ],
        project_dir,
    )
    print(f"completed final model; results={results_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
