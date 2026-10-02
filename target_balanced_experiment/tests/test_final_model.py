import unittest

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from chemwave_multitask import TargetConditionedChemWave
from chemwave_training import fit_stage, make_target_balanced_loader


class FinalModelTests(unittest.TestCase):
    def test_target_balanced_sampler_equalizes_expected_target_mass(self):
        graphs = []
        for target_id, count in ((0, 2), (1, 8), (2, 20)):
            for _ in range(count):
                graphs.append(
                    Data(
                        x=torch.zeros(1, 35),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty((0, 12)),
                        relative_position=torch.zeros(1, 45),
                        target_id=torch.tensor([target_id]),
                        y=torch.tensor([0.0]),
                    )
                )
        loader = make_target_balanced_loader(graphs, batch_size=5, seed=7)
        sampler_weights = loader.sampler.weights
        target_mass = []
        offset = 0
        for count in (2, 8, 20):
            target_mass.append(float(sampler_weights[offset : offset + count].sum()))
            offset += count
        self.assertTrue(torch.allclose(torch.tensor(target_mass), torch.ones(3)))

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

        inactive = torch.tensor([0, 2])
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
