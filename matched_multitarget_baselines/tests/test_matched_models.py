from __future__ import annotations

import copy
import unittest

import torch
from torch_geometric.data import Batch

from chemwave_features import molecule_to_graph35
from matched_models import build_model
from matched_training import fit_stage, make_loader


class MatchedModelTests(unittest.TestCase):
    def setUp(self):
        self.graphs = [
            molecule_to_graph35(
                "CC(=O)Nc1ccc(Cl)cc1", 6.0, target_index=0
            ),
            molecule_to_graph35("CCOc1ccccc1", 5.5, target_index=1),
        ]
        self.batch = Batch.from_data_list(self.graphs)

    def test_both_backbones_forward_and_backward(self):
        for model_name in ("gine", "attentivefp"):
            with self.subTest(model_name=model_name):
                torch.manual_seed(4)
                model = build_model(model_name, num_targets=2)
                prediction = model(self.batch)
                self.assertEqual(prediction.shape, (2,))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.backbone.parameters()
                    )
                )

    def test_target_adaptation_budget_is_exactly_2182(self):
        for model_name in ("gine", "attentivefp"):
            model = build_model(model_name, num_targets=30)
            self.assertEqual(model.active_target_coordinate_count(), 2182)
            row_count = sum(
                parameter[0].numel()
                for parameter in model.target_row_parameters()
            )
            self.assertEqual(row_count, 2182)

    def test_relative_position_field_is_not_read(self):
        for model_name in ("gine", "attentivefp"):
            with self.subTest(model_name=model_name):
                torch.manual_seed(8)
                model = build_model(model_name, num_targets=2)
                model.eval()
                altered = [copy.deepcopy(graph) for graph in self.graphs]
                for graph in altered:
                    graph.relative_position = torch.randn_like(
                        graph.relative_position
                    )
                with torch.no_grad():
                    original = model(self.batch)
                    changed = model(Batch.from_data_list(altered))
                self.assertTrue(torch.equal(original, changed))

    def test_adaptation_changes_only_the_addressed_target_rows(self):
        torch.manual_seed(12)
        model = build_model("gine", num_targets=2)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.target_row_parameters():
            parameter.requires_grad = True
        with torch.no_grad():
            for parameter in model.target_row_parameters():
                parameter[1] = 1.0e8
        initial = [
            parameter.detach().clone()
            for parameter in model.target_row_parameters()
        ]
        target_graphs = [
            molecule_to_graph35("CCO", 5.0, target_index=0),
            molecule_to_graph35("CCN", 6.0, target_index=0),
        ]
        fit_stage(
            model,
            make_loader(target_graphs, 2, True, 0),
            make_loader(target_graphs, 2, False, 0),
            lr=1e-4,
            max_epochs=1,
            device=torch.device("cpu"),
            frozen_backbone=True,
            active_target_index=0,
        )
        changed_active = False
        for parameter, before in zip(model.target_row_parameters(), initial):
            self.assertTrue(torch.equal(parameter.detach()[1], before[1]))
            changed_active |= not torch.equal(parameter.detach()[0], before[0])
        self.assertTrue(changed_active)


if __name__ == "__main__":
    unittest.main()
