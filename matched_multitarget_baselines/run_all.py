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
            "Run one five-seed matched multi-target baseline, followed by "
            "target adaptation and official testing on all 30 datasets."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("matched_runs"))
    parser.add_argument(
        "--model-name", choices=["gine", "attentivefp"], required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--expected-target-count", type=int, default=30)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / args.model_name
    pretrain_dir = model_dir / "checkpoints" / "pretrain"
    adapted_dir = model_dir / "checkpoints" / "adapted"
    results_dir = model_dir / "results"
    seeds = [str(seed) for seed in DEFAULT_SEEDS]

    common = [
        "--data-dir",
        str(data_dir),
        "--pretrain-dir",
        str(pretrain_dir),
        "--model-name",
        args.model_name,
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
            str(project_dir / "run_pretrain.py"),
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
            str(project_dir / "run_finetune_test.py"),
            *common,
            "--adapted-dir",
            str(adapted_dir),
            "--results-csv",
            str(results_dir / "all_results.csv"),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--expected-target-count",
            str(args.expected_target_count),
        ],
        project_dir,
    )
    print(
        f"completed matched {args.model_name}; "
        f"results={results_dir / 'all_results.csv'}"
    )


if __name__ == "__main__":
    main()
