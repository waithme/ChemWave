from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from chemwave_features import molecule_to_graph35
from matched_models import build_model


DATA_DIR = Path("data/experiment2")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def prepare_development_data(
    target_name: str | None = None, data_dir: Path = DATA_DIR
):
    """Build train/validation graphs without loading test activities or cliff labels."""
    paths = sorted(data_dir.glob("CHEMBL*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CHEMBL*.csv files found under {data_dir}")
    target_names = [path.stem for path in paths]
    if target_name is not None and target_name not in target_names:
        raise ValueError(f"Unknown target: {target_name}")
    target_to_index = {name: index for index, name in enumerate(target_names)}
    train_frames = {}
    test_identities = set()
    for path in paths:
        # Pass 1 reads only structure and split membership. Official-test activities
        # and cliff annotations are never loaded by the development process.
        split_frame = pd.read_csv(path, usecols=["smiles", "split"])
        split_frame["split"] = split_frame["split"].str.lower()
        split_frame["canonical"] = [
            canonical_smiles(value) for value in split_frame["smiles"]
        ]
        test_identities.update(
            split_frame.loc[split_frame["split"] == "test", "canonical"].dropna()
        )

        # Pass 2 physically skips all non-train CSV rows before parsing activities.
        train_rows = set(split_frame.index[split_frame["split"] == "train"].tolist())
        frame = pd.read_csv(
            path,
            usecols=["smiles", "y [pEC50/pKi]"],
            skiprows=lambda line_number: (
                line_number > 0 and (line_number - 1) not in train_rows
            ),
        )
        frame["canonical"] = [canonical_smiles(value) for value in frame["smiles"]]
        train_frames[path.stem] = frame.dropna(
            subset=["canonical", "y [pEC50/pKi]"]
        ).copy()

    retained_by_target = {}
    for current_name, frame in train_frames.items():
        retained_by_target[current_name] = frame[
            ~frame["canonical"].isin(test_identities)
        ].copy()

    pretrain_graphs, validation_graphs = [], []
    target_train, target_val = [], []
    target_index = target_to_index[target_name] if target_name is not None else None
    identity_counts = {}
    target_totals = np.zeros(len(target_names), dtype=np.int64)
    for current_name, retained in retained_by_target.items():
        current_target = target_to_index[current_name]
        for canonical in retained["canonical"]:
            identity_counts.setdefault(
                canonical, np.zeros(len(target_names), dtype=np.int64)
            )[current_target] += 1
            target_totals[current_target] += 1

    # One global canonical-identity assignment, while retaining exactly 10% of
    # each target's filtered rows as audited in Supporting Information Table S4.
    validation_quota = np.ceil(target_totals * 0.10).astype(np.int64)
    validation_counts = np.zeros_like(validation_quota)
    shuffled_identities = sorted(identity_counts)
    np.random.RandomState(0).shuffle(shuffled_identities)
    validation_identities = set()
    for canonical in shuffled_identities:
        contribution = identity_counts[canonical]
        if np.all(validation_counts + contribution <= validation_quota):
            validation_identities.add(canonical)
            validation_counts += contribution
    if not np.array_equal(validation_counts, validation_quota):
        raise RuntimeError("Unable to construct the audited global identity split")

    for current_name, retained in retained_by_target.items():
        for canonical, smiles, y in zip(
            retained["canonical"],
            retained["smiles"],
            retained["y [pEC50/pKi]"],
        ):
            graph = molecule_to_graph35(
                smiles, y, target_index=target_to_index[current_name]
            )
            if "cliff_label" in graph.keys():
                raise AssertionError("Development graph contains a cliff label")
            if canonical in validation_identities:
                validation_graphs.append(graph)
                if target_name is not None and current_name == target_name:
                    target_val.append(graph)
            else:
                pretrain_graphs.append(graph)
                if target_name is not None and current_name == target_name:
                    target_train.append(graph)

    pretrain_identities = set(identity_counts) - validation_identities
    identity_overlap = pretrain_identities & validation_identities
    retained_identities = set(identity_counts)
    official_test_overlap = retained_identities & test_identities
    if identity_overlap:
        raise AssertionError("Canonical identity overlap between train and validation")
    if official_test_overlap:
        raise AssertionError("Official-test identity reached development data")

    fingerprint = hashlib.sha256()
    for current_name in target_names:
        retained = retained_by_target[current_name]
        for canonical, y in zip(
            retained["canonical"], retained["y [pEC50/pKi]"]
        ):
            partition = "validation" if canonical in validation_identities else "train"
            record = [current_name, canonical, float(y), partition]
            fingerprint.update(
                json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
            )
    return {
        "target_names": target_names,
        "target_index": target_index,
        "pretrain": pretrain_graphs,
        "pretrain_val": validation_graphs,
        "target_train": target_train,
        "target_val": target_val,
        "development_data_sha256": fingerprint.hexdigest(),
        "identity_audit": {
            "canonical_pretrain_validation_overlap": len(identity_overlap),
            "canonical_development_official_test_overlap": len(
                official_test_overlap
            ),
        },
    }


def select_target_data(data: dict, target_name: str) -> dict:
    if target_name not in data["target_names"]:
        raise ValueError(f"Unknown target: {target_name}")
    target_index = data["target_names"].index(target_name)
    selected = dict(data)
    selected["target_index"] = target_index
    selected["target_train"] = [
        graph
        for graph in data["pretrain"]
        if int(graph.target_id.item()) == target_index
    ]
    selected["target_val"] = [
        graph
        for graph in data["pretrain_val"]
        if int(graph.target_id.item()) == target_index
    ]
    if not selected["target_train"] or not selected["target_val"]:
        raise RuntimeError(f"Empty train/validation partition for {target_name}")
    return selected


@torch.no_grad()
def validation_mse(model, loader, device):
    model.eval()
    squared_error, count = 0.0, 0
    for batch in loader:
        if "cliff_label" in batch.keys():
            raise AssertionError("Validation batch contains a cliff label")
        batch = batch.to(device)
        squared_error += F.mse_loss(model(batch), batch.y.view(-1), reduction="sum").item()
        count += batch.num_graphs
    return squared_error / count


def make_loader(graphs, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def initialize_output_bias(model, graphs, num_targets):
    sums = torch.zeros(num_targets)
    counts = torch.zeros(num_targets)
    for graph in graphs:
        target = int(graph.target_id.item())
        sums[target] += graph.y.item()
        counts[target] += 1
    means = sums / counts.clamp_min(1)
    model.output_bias.data.copy_(means.to(model.output_bias.device))


def _trainable_target_rows(model):
    """Return target-indexed tensors whose inactive rows must stay frozen."""
    return [
        parameter
        for parameter in model.target_row_parameters()
        if parameter.requires_grad
    ]


def fit_stage(
    model,
    train_loader,
    val_loader,
    lr,
    max_epochs,
    device,
    frozen_backbone=False,
    stage_name=None,
    active_target_index=None,
):
    protected_rows = []
    inactive_mask = None
    if active_target_index is not None:
        target_parameters = _trainable_target_rows(model)
        if not target_parameters:
            raise RuntimeError("Adaptation has no trainable target-indexed parameters")
        num_targets = target_parameters[0].shape[0]
        if not 0 <= active_target_index < num_targets:
            raise IndexError(f"Invalid active target index: {active_target_index}")
        inactive_mask = torch.ones(
            num_targets, dtype=torch.bool, device=target_parameters[0].device
        )
        inactive_mask[active_target_index] = False
        for parameter in target_parameters:
            if parameter.shape[0] != num_targets:
                raise RuntimeError("Inconsistent target dimension during adaptation")
            protected_rows.append((parameter, parameter.detach().clone()))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=1e-5,
    )
    best_state, best_mse, best_epoch, stale = None, math.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if frozen_backbone:
            model.freeze_shared_modules_eval()
        for batch in train_loader:
            if "cliff_label" in batch.keys():
                raise AssertionError("Training batch contains a cliff label")
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if protected_rows:
                # AdamW applies decoupled weight decay to the whole dense
                # tensor, including rows with zero data gradient. Restore all
                # inactive rows after every step so only the addressed target
                # can change while preserving AdamW on its active row.
                with torch.no_grad():
                    for parameter, initial_value in protected_rows:
                        parameter[inactive_mask] = initial_value[inactive_mask]
        mse = validation_mse(model, val_loader, device)
        improved = mse < best_mse - 1e-8
        if improved:
            best_state, best_mse, best_epoch, stale = copy.deepcopy(model.state_dict()), mse, epoch, 0
        else:
            stale += 1
        if stage_name is not None:
            print(
                json.dumps(
                    {
                        "stage": stage_name,
                        "epoch": epoch,
                        "validation_rmse": math.sqrt(mse),
                        "best_epoch": best_epoch,
                        "best_validation_rmse": math.sqrt(best_mse),
                        "improved": improved,
                        "stale_epochs": stale,
                        "epoch_seconds": time.perf_counter() - epoch_start,
                    }
                ),
                flush=True,
            )
        if stale >= 15:
            break
    model.load_state_dict(best_state)
    if protected_rows:
        for parameter, initial_value in protected_rows:
            if not torch.equal(
                parameter.detach()[inactive_mask], initial_value[inactive_mask]
            ):
                raise AssertionError(
                    "Inactive target rows changed during target adaptation"
                )
    return best_epoch, math.sqrt(best_mse)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_checkpoint(
    checkpoint: dict,
    *,
    stage: str,
    model_name: str,
    seed: int,
    data: dict,
    target_name: str | None = None,
) -> None:
    expected = {
        "stage": stage,
        "model_name": model_name,
        "development_data_sha256": data["development_data_sha256"],
        "target_names": data["target_names"],
        "seed": seed,
    }
    if target_name is not None:
        expected["target_name"] = target_name
    mismatches = {
        key: {"expected": value, "found": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint metadata mismatch:\n"
            + json.dumps(mismatches, indent=2, default=str)
        )
    if "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint has no model_state")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def ensure_shared_pretrain(data, seed, args, device) -> tuple[dict, str, Path]:
    data = {**data, "model_name": args.model_name}
    path = args.pretrain_dir / f"{args.model_name}_shared_pretrain_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    print(f"checking shared pretrain checkpoint: {path}", flush=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists():
            checkpoint = load_checkpoint(path, device)
            validate_checkpoint(
                checkpoint,
                stage="shared_pretrain",
                model_name=args.model_name,
                seed=seed,
                data=data,
            )
            print(f"reusing shared pretrain checkpoint: {path}", flush=True)
            return checkpoint, "reused", path

        print(f"training missing shared pretrain checkpoint: {path}", flush=True)
        seed_everything(seed)
        model = build_model(
            args.model_name,
            num_targets=len(data["target_names"]),
            hidden_dim=args.hidden_dim,
        ).to(device)
        initialize_output_bias(model, data["pretrain"], len(data["target_names"]))
        best_epoch, validation_rmse = fit_stage(
            model,
            make_loader(data["pretrain"], 128, True, seed),
            make_loader(data["pretrain_val"], 128, False, seed),
            1e-3,
            args.pretrain_epochs,
            device,
            stage_name=f"shared_pretrain_seed_{seed}",
        )
        checkpoint = {
            "stage": "shared_pretrain",
            "model_name": args.model_name,
            "development_data_sha256": data["development_data_sha256"],
            "target_names": data["target_names"],
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_rmse": validation_rmse,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "active_target_coordinates": model.active_target_coordinate_count(),
            "model_state": model.state_dict(),
        }
        atomic_torch_save(checkpoint, path)
        return checkpoint, "trained", path


def ensure_target_adaptation(
    pretrain_checkpoint, data, seed, args, device
) -> tuple[dict, str, Path]:
    data = {**data, "model_name": args.model_name}
    path = (
        args.adapted_dir
        / args.target_name
        / f"{args.model_name}_adapted_seed{seed}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    print(f"checking target adaptation checkpoint: {path}", flush=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists():
            checkpoint = load_checkpoint(path, device)
            validate_checkpoint(
                checkpoint,
                stage="adapted",
                model_name=args.model_name,
                seed=seed,
                data=data,
                target_name=args.target_name,
            )
            print(f"reusing target adaptation checkpoint: {path}", flush=True)
            return checkpoint, "reused", path

        print(f"training missing target adaptation checkpoint: {path}", flush=True)
        seed_everything(seed)
        model = build_model(
            args.model_name,
            num_targets=len(data["target_names"]),
            hidden_dim=args.hidden_dim,
        ).to(device)
        model.load_state_dict(pretrain_checkpoint["model_state"])
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.target_row_parameters():
            parameter.requires_grad = True
        best_epoch, validation_rmse = fit_stage(
            model,
            make_loader(data["target_train"], 128, True, seed),
            make_loader(data["target_val"], 128, False, seed),
            1e-4,
            args.finetune_epochs,
            device,
            frozen_backbone=True,
            stage_name=f"adapt_{args.target_name}_seed_{seed}",
            active_target_index=data["target_names"].index(args.target_name),
        )
        checkpoint = {
            "stage": "adapted",
            "model_name": args.model_name,
            "development_data_sha256": data["development_data_sha256"],
            "target_names": data["target_names"],
            "target_name": args.target_name,
            "seed": seed,
            "shared_pretrain_checkpoint": str(
                args.pretrain_dir
                / f"{args.model_name}_shared_pretrain_seed{seed}.pt"
            ),
            "best_epoch": best_epoch,
            "validation_rmse": validation_rmse,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "active_target_coordinates": model.active_target_coordinate_count(),
            "model_state": model.state_dict(),
        }
        atomic_torch_save(checkpoint, path)
        return checkpoint, "trained", path
