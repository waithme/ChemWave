from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_experiment3_evaluate import PROTOCOLS


def run(command: list[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final-model training-strategy experiment: from scratch, "
            "joint only, and joint plus target adaptation."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment3_runs"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS)
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Optional subset; default is all 30 datasets."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--scratch-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--expected-target-count", type=int, default=30)
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if len(set(args.protocols)) != len(args.protocols):
        raise ValueError("Duplicate protocols are not allowed")
    project_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    pretrain_dir = output_dir / "checkpoints" / "joint_pretrain"
    adapted_dir = output_dir / "checkpoints" / "joint_adaptation"
    scratch_dir = output_dir / "checkpoints" / "from_scratch"
    results_dir = output_dir / "results"
    seed_args = [str(seed) for seed in args.seeds]

    if any(protocol != "from_scratch" for protocol in args.protocols):
        run(
            [
                sys.executable,
                str(project_dir / "run_chemwave_pretrain.py"),
                "--data-dir",
                str(data_dir),
                "--pretrain-dir",
                str(pretrain_dir),
                "--seeds",
                *seed_args,
                "--hidden-dim",
                str(args.hidden_dim),
                "--pretrain-epochs",
                str(args.pretrain_epochs),
                "--device",
                args.device,
                "--output",
                str(results_dir / "joint_pretrain.json"),
            ],
            project_dir,
        )

    command = [
        sys.executable,
        str(project_dir / "run_experiment3_evaluate.py"),
        "--data-dir",
        str(data_dir),
        "--pretrain-dir",
        str(pretrain_dir),
        "--adapted-dir",
        str(adapted_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--results-csv",
        str(results_dir / "all_results.csv"),
        "--seeds",
        *seed_args,
        "--protocols",
        *args.protocols,
        "--hidden-dim",
        str(args.hidden_dim),
        "--scratch-epochs",
        str(args.scratch_epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--expected-target-count",
        str(args.expected_target_count),
        "--device",
        args.device,
    ]
    if args.targets is not None:
        command.extend(["--targets", *args.targets])
    run(command, project_dir)
    print(f"completed experiment3; results={results_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
