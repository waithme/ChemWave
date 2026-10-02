from __future__ import annotations

import copy
import unittest

import torch
from torch_geometric.data import Batch

from chemwave_features import molecule_to_graph35
from matched_models import build_model
from matched_training import fit_stage, make_loader


class GINEFiLMControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = [
            molecule_to_graph35("CC(=O)Nc1ccc(Cl)cc1", 6.0, target_index=0),
            molecule_to_graph35("CCOc1ccccc1", 5.5, target_index=1),
        ]
        self.batch = Batch.from_data_list(self.graphs)

    def test_same_parameter_count_and_active_budget_as_matched_gine(self) -> None:
        standard = build_model("gine", num_targets=30)
        film = build_model("gine_film", num_targets=30)
        self.assertEqual(
            sum(p.numel() for p in standard.parameters()),
            sum(p.numel() for p in film.parameters()),
        )
        self.assertEqual(film.active_target_coordinate_count(), 2182)
        self.assertEqual(
            sum(p[0].numel() for p in film.target_row_parameters()), 2182
        )

    def test_zero_film_matches_postpool_gine_and_conditioning_enters_graph(self) -> None:
        torch.manual_seed(4)
        standard = build_model("gine", num_targets=2).eval()
        film = build_model("gine_film", num_targets=2).eval()
        film.load_state_dict(standard.state_dict())
        with torch.no_grad():
            expected = standard(self.batch)
            self.assertTrue(torch.allclose(film(self.batch), expected, atol=1e-7))
            before = film.backbone(
                self.batch,
                film.target_projection(film.target_embedding(self.batch.target_id)),
                film.target_scale(self.batch.target_id).view(2, 3, 300),
                film.target_shift(self.batch.target_id).view(2, 3, 300),
            )
            film.target_scale.weight[0].fill_(0.4)
            film.target_shift.weight[0].fill_(0.1)
            after = film.backbone(
                self.batch,
                film.target_projection(film.target_embedding(self.batch.target_id)),
                film.target_scale(self.batch.target_id).view(2, 3, 300),
                film.target_shift(self.batch.target_id).view(2, 3, 300),
            )
        self.assertFalse(torch.allclose(before[0], after[0]))
        self.assertTrue(torch.allclose(before[1], after[1], atol=1e-7))

    def test_relative_position_field_is_not_read(self) -> None:
        torch.manual_seed(8)
        model = build_model("gine_film", num_targets=2).eval()
        altered = [copy.deepcopy(graph) for graph in self.graphs]
        for graph in altered:
            graph.relative_position = torch.randn_like(graph.relative_position)
        with torch.no_grad():
            original = model(self.batch)
            changed = model(Batch.from_data_list(altered))
        self.assertTrue(torch.equal(original, changed))

    def test_adaptation_changes_only_the_addressed_assay_rows(self) -> None:
        torch.manual_seed(12)
        model = build_model("gine_film", num_targets=2)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.target_row_parameters():
            parameter.requires_grad = True
        with torch.no_grad():
            for parameter in model.target_row_parameters():
                parameter[1] = 1.0e8
        initial = [p.detach().clone() for p in model.target_row_parameters()]
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
        self.assertTrue(
            any(
                not torch.equal(p.detach()[0], before[0])
                for p, before in zip(model.target_row_parameters(), initial)
            )
        )
        for parameter, before in zip(model.target_row_parameters(), initial):
            self.assertTrue(torch.equal(parameter.detach()[1], before[1]))


if __name__ == "__main__":
    unittest.main()
