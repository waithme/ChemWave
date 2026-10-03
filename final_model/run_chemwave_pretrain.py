from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chemwave_training import (
    ensure_shared_pretrain,
    prepare_development_data,
)
from chemwave_provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain one shared 30-target ChemWave model per seed."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    device = torch.device(args.device)
    data = prepare_development_data(data_dir=args.data_dir)
    counts = {
        "pretrain": len(data["pretrain"]),
        "pretrain_val": len(data["pretrain_val"]),
    }
    print(
        f"device={device} targets={len(data['target_names'])} counts={counts}",
        flush=True,
    )

    runs = []
    for seed in args.seeds:
        checkpoint, source, path = ensure_shared_pretrain(data, seed, args, device)
        run = {
            "seed": seed,
            "source": source,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "protocol_sha256": checkpoint["protocol_sha256"],
            "best_epoch": int(checkpoint["best_epoch"]),
            "validation_rmse": float(checkpoint["validation_rmse"]),
        }
        runs.append(run)
        print(json.dumps(run), flush=True)

    values = np.asarray([run["validation_rmse"] for run in runs])
    payload = {
        "stage": "shared_pretrain",
        "development_data_sha256": data["development_data_sha256"],
        "data_dir": str(args.data_dir),
        "target_names": data["target_names"],
        "counts": counts,
        "identity_audit": data["identity_audit"],
        "runs": runs,
        "summary": {
            "pretrain_val_rmse": {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "sample_variance": (
                    float(values.var(ddof=1)) if len(values) > 1 else None
                ),
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
