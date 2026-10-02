import unittest

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from chemwave_multitask import TargetConditionedChemWave
from chemwave_training import fit_stage


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

    def test_adaptation_preserves_inactive_target_rows(self):
        graphs = []
        for value in (0.5, 1.0):
            graphs.append(
                Data(
                    x=torch.randn(3, 35),
                    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                    edge_attr=torch.randn(4, 12),
                    relative_position=torch.randn(3, 45),
                    target_id=torch.tensor([1]),
                    y=torch.tensor([value]),
                )
            )
        model = TargetConditionedChemWave(num_targets=3, hidden_dim=16)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.blocks:
            block.target_frequency.weight.requires_grad = True
            block.target_bond_conductance.weight.requires_grad = True
            block.target_gradient_gain.weight.requires_grad = True
        model.target_relative_position.weight.requires_grad = True
        model.output_weight.requires_grad = True
        model.output_bias.requires_grad = True

        # Make decoupled weight decay observable in float32. Without a true
        # indexed write-back, the inactive rows would change during AdamW.
        inactive = torch.tensor([0, 2])
        with torch.no_grad():
            for parameter in (
                model.output_weight,
                model.output_bias,
                model.target_relative_position.weight,
                *[
                    embedding.weight
                    for block in model.blocks
                    for embedding in (
                        block.target_frequency,
                        block.target_bond_conductance,
                        block.target_gradient_gain,
                    )
                ],
            ):
                parameter[inactive] = 1.0e8

        target_parameters = [
            parameter.detach().clone()
            for parameter in (
                model.output_weight,
                model.output_bias,
                model.target_relative_position.weight,
                *[
                    embedding.weight
                    for block in model.blocks
                    for embedding in (
                        block.target_frequency,
                        block.target_bond_conductance,
                        block.target_gradient_gain,
                    )
                ],
            )
        ]
        loader = DataLoader(graphs, batch_size=2, shuffle=False)
        fit_stage(
            model,
            loader,
            loader,
            lr=1e-4,
            max_epochs=2,
            device=torch.device("cpu"),
            frozen_backbone=True,
            active_target_index=1,
        )

        current = (
            model.output_weight,
            model.output_bias,
            model.target_relative_position.weight,
            *[
                embedding.weight
                for block in model.blocks
                for embedding in (
                    block.target_frequency,
                    block.target_bond_conductance,
                    block.target_gradient_gain,
                )
            ],
        )
        for parameter, initial in zip(current, target_parameters):
            self.assertTrue(torch.equal(parameter.detach()[inactive], initial[inactive]))


if __name__ == "__main__":
    unittest.main()
