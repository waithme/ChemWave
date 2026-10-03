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
from run_interpretability import load_model
from chemwave_provenance import VARIANT, canonical_sha256, protocol, sha256_file
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

    def test_explanation_checkpoint_requires_matching_fingerprint(self):
        data = {"development_data_sha256": "dev-hash", "target_names": ["A", "B"]}
        built = protocol(
            stage="adapted", data=data, seed=0, hidden_dim=16, max_epochs=100,
            target_name="A", pretrain_checkpoint_sha256="a" * 64,
            pretrain_protocol_sha256="b" * 64,
        )
        payload = {
            "stage": "adapted", "variant": VARIANT, "target_name": "A",
            "target_names": ["A", "B"], "seed": 0,
            "development_data_sha256": "dev-hash",
            "protocol": built, "protocol_sha256": canonical_sha256(built),
            "model_state": self.model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            load_model(
                path, target="A", seed=0, device=torch.device("cpu"),
                expected_fingerprint="dev-hash", expected_target_names=["A", "B"],
            )
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                load_model(
                    path, target="A", seed=0, device=torch.device("cpu"),
                    expected_fingerprint="other", expected_target_names=["A", "B"],
                )
            payload.pop("protocol")
            payload.pop("protocol_sha256")
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Legacy checkpoint requires"):
                load_model(
                    path, target="A", seed=0, device=torch.device("cpu"),
                    expected_fingerprint="dev-hash", expected_target_names=["A", "B"],
                )
            load_model(
                path, target="A", seed=0, device=torch.device("cpu"),
                expected_fingerprint="dev-hash", expected_target_names=["A", "B"],
                legacy_sha256=sha256_file(path),
            )


if __name__ == "__main__":
    unittest.main()
