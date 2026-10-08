"""Stable identities for new ChemWave runs; never retroactively certify old weights."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy
import pandas
import rdkit
import torch
import torch_geometric


# Historical protocol identity of final Full (canonical ablation name:
# a5_full). Keep this value stable for checkpoint/result provenance validation.
VARIANT = "a5_plain_bond_gradient"
PROTOCOL_VERSION = 2
MODEL_OPTIONS = {
    "atom_dim": 35,
    "relative_position_dim": 45,
    "bond_dim": 12,
    "num_layers": 3,
    "dropout": 0.1,
}
TRAIN_OPTIONS = {
    "batch_size": 128,
    "optimizer": "AdamW",
    "weight_decay": 1e-5,
    "gradient_clip_norm": 5.0,
    "early_stop_patience": 15,
    "minimum_mse_improvement": 1e-8,
    "validation_metric": "MSE (selection), RMSE (reporting)",
    "split": "global canonical identity, 10% target quota, seed 0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def code_hashes(directory: Path | None = None) -> dict[str, str]:
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    return {
        name: sha256_file(directory / name)
        for name in ("chemwave_features.py", "chemwave_multitask.py", "chemwave_training.py")
    }


def protocol(
    *,
    stage: str,
    data: dict,
    seed: int,
    hidden_dim: int,
    max_epochs: int,
    code: dict[str, str] | None = None,
    target_name: str | None = None,
    pretrain_checkpoint_sha256: str | None = None,
    pretrain_protocol_sha256: str | None = None,
) -> dict:
    if stage not in ("shared_pretrain", "adapted"):
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "adapted" and not (target_name and pretrain_checkpoint_sha256 and pretrain_protocol_sha256):
        raise ValueError("Adaptation requires target and exact pretrain provenance")
    if stage == "shared_pretrain" and any((target_name, pretrain_checkpoint_sha256, pretrain_protocol_sha256)):
        raise ValueError("Pretraining cannot include adaptation fields")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": VARIANT,
        "stage": stage,
        "development_data_sha256": data["development_data_sha256"],
        "target_names": list(data["target_names"]),
        "seed": int(seed),
        "model": {**MODEL_OPTIONS, "hidden_dim": int(hidden_dim), "num_targets": len(data["target_names"])},
        "training": {
            **TRAIN_OPTIONS,
            "max_epochs": int(max_epochs),
            "learning_rate": 1e-3 if stage == "shared_pretrain" else 1e-4,
            "row_weighting": "training rows",
            "frozen_backbone": stage == "adapted",
            "inactive_target_rows_restored": stage == "adapted",
        },
        "source_sha256": code if code is not None else code_hashes(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "rdkit": rdkit.__version__,
            "torch": str(torch.__version__),
            "torch_geometric": torch_geometric.__version__,
        },
    }
    if stage == "adapted":
        payload.update(
            target_name=target_name,
            pretrain_checkpoint_sha256=pretrain_checkpoint_sha256,
            pretrain_protocol_sha256=pretrain_protocol_sha256,
        )
    return payload


def validate_protocol(checkpoint: dict, expected: dict) -> str:
    if "protocol" not in checkpoint or "protocol_sha256" not in checkpoint:
        raise RuntimeError(
            "Legacy checkpoint has no configuration fingerprint; it cannot be "
            "silently reused. Keep it unchanged and use a fresh run directory."
        )
    stored = checkpoint["protocol"]
    if checkpoint["protocol_sha256"] != canonical_sha256(stored):
        raise RuntimeError("Checkpoint's stored protocol fingerprint is corrupt")
    expected_hash = canonical_sha256(expected)
    if checkpoint["protocol_sha256"] != expected_hash or stored != expected:
        raise RuntimeError(
            "Checkpoint configuration/source mismatch; refusing reuse. "
            f"expected={expected_hash}, found={checkpoint['protocol_sha256']}"
        )
    return expected_hash
