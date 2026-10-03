import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from chemwave_provenance import canonical_sha256, protocol, validate_protocol
from chemwave_training import ensure_shared_pretrain, ensure_target_adaptation, select_target_data
from run_chemwave_finetune_test import CSV_FIELDS, append_csv_once, csv_contains


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "development_data_sha256": "development-hash",
            "target_names": ["CHEMBL_A", "CHEMBL_B"],
        }
        self.code = {"model": "model-hash", "features": "features-hash", "training": "training-hash"}

    def test_same_shape_different_training_config_is_rejected(self):
        expected = protocol(
            stage="shared_pretrain", data=self.data, seed=0,
            hidden_dim=16, max_epochs=100, code=self.code,
        )
        checkpoint = {"protocol": expected, "protocol_sha256": canonical_sha256(expected)}
        self.assertEqual(validate_protocol(checkpoint, expected), checkpoint["protocol_sha256"])
        changed = protocol(
            stage="shared_pretrain", data=self.data, seed=0,
            hidden_dim=16, max_epochs=99, code=self.code,
        )
        with self.assertRaisesRegex(RuntimeError, "configuration/source mismatch"):
            validate_protocol(checkpoint, changed)
        with self.assertRaisesRegex(RuntimeError, "Legacy checkpoint"):
            validate_protocol({}, expected)

    def test_adaptation_binds_exact_pretrain_bytes(self):
        args = dict(
            stage="adapted", data=self.data, seed=0, hidden_dim=16,
            max_epochs=100, target_name="CHEMBL_A",
            pretrain_protocol_sha256="protocol-hash", code=self.code,
        )
        first = protocol(**args, pretrain_checkpoint_sha256="a" * 64)
        second = protocol(**args, pretrain_checkpoint_sha256="b" * 64)
        self.assertNotEqual(canonical_sha256(first), canonical_sha256(second))

    def test_result_skip_requires_identical_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            row = {field: "value" for field in CSV_FIELDS}
            row.update(
                development_data_sha256="dev", target="CHEMBL_A", seed=0,
                adapted_checkpoint_sha256="a" * 64,
            )
            self.assertTrue(append_csv_once(path, row))
            self.assertTrue(csv_contains(path, row))
            self.assertFalse(append_csv_once(path, row))
            changed = dict(row, adapted_checkpoint_sha256="b" * 64)
            with self.assertRaisesRegex(RuntimeError, "differs"):
                csv_contains(path, changed)
            with self.assertRaisesRegex(RuntimeError, "differs"):
                append_csv_once(path, changed)
            with path.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_legacy_result_table_is_not_upgraded_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("development_data_sha256,target,seed\ndev,CHEMBL_A,0\n")
            row = {field: "value" for field in CSV_FIELDS}
            with self.assertRaisesRegex(RuntimeError, "Preserve the historical table"):
                csv_contains(path, row)

    def test_tiny_training_run_writes_and_checks_provenance(self):
        def graph(target, value):
            return Data(
                x=torch.randn(3, 35),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                edge_attr=torch.randn(4, 12),
                relative_position=torch.randn(3, 45),
                target_id=torch.tensor([target]),
                y=torch.tensor([value]),
            )

        data = {
            "development_data_sha256": "test-development",
            "target_names": ["CHEMBL_A", "CHEMBL_B"],
            "pretrain": [graph(0, 1.0), graph(0, 1.1), graph(1, 2.0), graph(1, 2.1)],
            "pretrain_val": [graph(0, 1.2), graph(1, 2.2)],
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                pretrain_dir=Path(directory) / "pretrain",
                adapted_dir=Path(directory) / "adapted",
                hidden_dim=16, pretrain_epochs=1, finetune_epochs=1,
                target_name="CHEMBL_A",
            )
            device = torch.device("cpu")
            pretrain, source, _ = ensure_shared_pretrain(data, 0, args, device)
            self.assertEqual(source, "trained")
            self.assertEqual(ensure_shared_pretrain(data, 0, args, device)[1], "reused")
            selected = select_target_data(data, args.target_name)
            adapted, source, _ = ensure_target_adaptation(pretrain, selected, 0, args, device)
            self.assertEqual(source, "trained")
            self.assertEqual(ensure_target_adaptation(pretrain, selected, 0, args, device)[1], "reused")
            self.assertEqual(
                adapted["protocol"]["pretrain_protocol_sha256"],
                pretrain["protocol_sha256"],
            )
            args.finetune_epochs = 2
            with self.assertRaisesRegex(RuntimeError, "configuration/source mismatch"):
                ensure_target_adaptation(pretrain, selected, 0, args, device)


if __name__ == "__main__":
    unittest.main()
