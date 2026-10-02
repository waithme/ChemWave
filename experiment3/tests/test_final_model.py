import unittest

import torch
from torch_geometric.data import Batch, Data

from chemwave_multitask import TargetConditionedChemWave


class FinalModelTests(unittest.TestCase):
    def test_forward_and_backward(self):
        graph = Data(
            x=torch.randn(3, 35),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.randn(4, 12),
            relative_position=torch.randn(3, 45),
            target_id=torch.tensor([0]),
            y=torch.tensor([1.0]),
        )
        batch = Batch.from_data_list([graph])
        model = TargetConditionedChemWave(num_targets=2, hidden_dim=16)
        output = model(batch)
        output.square().mean().backward()
        self.assertEqual(tuple(output.shape), (1,))
        self.assertTrue(torch.isfinite(output).all())

    def test_no_bond_transport_parameters(self):
        model = TargetConditionedChemWave(num_targets=2, hidden_dim=16)
        names = set(dict(model.named_parameters()))
        self.assertFalse(any("bond_transport" in name for name in names))
        self.assertTrue(
            any("target_bond_conductance" in name for name in names)
        )
        self.assertTrue(any("target_gradient_gain" in name for name in names))


if __name__ == "__main__":
    unittest.main()
