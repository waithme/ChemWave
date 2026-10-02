import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from chemwave_explain import (
    explain_molecule,
    faithfulness_interventions,
    forward_with_interventions,
)
from chemwave_features import molecule_to_graph35
from chemwave_multitask import TargetConditionedChemWave
from interpretability_visuals import draw_molecule_explanation


class InterpretabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = molecule_to_graph35(
            "CC(=O)Nc1ccc(Cl)cc1", 6.0, target_index=0
        )
        torch.manual_seed(7)
        cls.model = TargetConditionedChemWave(
            num_targets=2, hidden_dim=16
        )
        cls.model.eval()

    def test_unmasked_explanation_forward_matches_model(self):
        batch = Batch.from_data_list([self.graph])
        with torch.no_grad():
            expected = self.model(batch)
            found = forward_with_interventions(self.model, batch)
        self.assertTrue(torch.equal(expected, found))

    def test_atom_bond_interventions_and_faithfulness(self):
        explanation = explain_molecule(self.model, self.graph)
        self.assertEqual(
            len(explanation["atom_prediction_delta"]), self.graph.num_nodes
        )
        self.assertEqual(len(explanation["bonds"]), self.graph.num_edges // 2)
        self.assertTrue(np.isfinite(explanation["atom_prediction_delta"]).all())
        records = faithfulness_interventions(
            self.model,
            self.graph,
            explanation,
            top_k=(1, 3),
            random_repeats=2,
        )
        self.assertEqual(len(records), 4)
        self.assertTrue(all(np.isfinite(row["top_change"]) for row in records))

    def test_rdkit_explanation_render(self):
        explanation = explain_molecule(self.model, self.graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "molecule.png"
            draw_molecule_explanation(
                self.graph.smiles,
                explanation["atom_prediction_delta"],
                explanation["bonds"],
                path,
                legend="test",
                width=500,
                height=350,
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
