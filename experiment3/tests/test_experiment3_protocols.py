import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from chemwave_training import ensure_target_from_scratch
from run_experiment3_evaluate import result_key


def tiny_graph(value: float) -> Data:
    return Data(
        x=torch.randn(3, 35),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 12),
        relative_position=torch.randn(3, 45),
        target_id=torch.tensor([0]),
        y=torch.tensor([value]),
    )


class Experiment3ProtocolTests(unittest.TestCase):
    def test_protocol_is_part_of_result_identity(self):
        base = {
            "development_data_sha256": "data",
            "target": "target",
            "seed": 0,
        }
        scratch = result_key({**base, "protocol": "from_scratch"})
        joint = result_key({**base, "protocol": "joint_only"})
        self.assertNotEqual(scratch, joint)

    def test_from_scratch_checkpoint_trains_and_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                scratch_dir=Path(directory),
                target_name="target_0",
                hidden_dim=16,
                scratch_epochs=1,
            )
            data = {
                "target_names": ["target_0"],
                "development_data_sha256": "tiny-data",
                "target_train": [tiny_graph(1.0), tiny_graph(1.2)],
                "target_val": [tiny_graph(1.1)],
            }
            trained, source, path = ensure_target_from_scratch(
                data, seed=0, args=args, device=torch.device("cpu")
            )
            self.assertEqual(source, "trained")
            self.assertEqual(trained["stage"], "from_scratch")
            self.assertTrue(path.exists())
            reused, source, reused_path = ensure_target_from_scratch(
                data, seed=0, args=args, device=torch.device("cpu")
            )
            self.assertEqual(source, "reused")
            self.assertEqual(path, reused_path)
            self.assertEqual(reused["target_name"], "target_0")


if __name__ == "__main__":
    unittest.main()
