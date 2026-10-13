import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from interpretability_visuals import _pyplot, plot_stability
from recompute_magnitude_seed_stability import recompute_saved_stability
from run_interpretability import signed_spearman


class AnalysisOutputTests(unittest.TestCase):
    def write_fixture(self, directory):
        tables = Path(directory)
        arrays = [np.array([(-1 if seed % 2 else 1) * 0.9, 0.4, 0.1])
                  for seed in range(5)]
        signed = np.mean([signed_spearman(a, b) for a, b in combinations(arrays, 2)])
        for filename, index, value in (
            ("atom_scores.csv", "atom_index", "relative_prediction_delta"),
            ("bond_scores.csv", "bond_index", "gradient_prediction_delta"),
        ):
            pd.DataFrame([
                {"target": "A", "row_id": 7, "seed": seed, index: component,
                 value: float(score)}
                for seed, scores in enumerate(arrays)
                for component, score in enumerate(scores)
            ]).to_csv(tables / filename, index=False)
        pd.DataFrame([{"target": "A", "row_id": 7,
                       "atom_spearman_mean": signed, "bond_spearman_mean": signed,
                       "atom_top3_overlap_mean": 1.0, "bond_top3_overlap_mean": 1.0}]
                     ).to_csv(tables / "seed_stability.csv", index=False)
        (tables / "seed_stability_magnitude_audit.json").write_text("historical audit\n")
        return tables

    def test_reaggregation_preserves_inputs_and_historical_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = self.write_fixture(directory)
            originals = {p: p.read_bytes() for p in tables.iterdir()}
            summary = recompute_saved_stability(tables, expected_molecules=1, expected_assays=1)
            output = tables / "seed_stability_magnitude.csv"
            self.assertTrue(output.is_file())
            self.assertAlmostEqual(pd.read_csv(output).atom_spearman_mean.iloc[0], 1.0)
            self.assertTrue(summary["source_tables_unchanged"])
            self.assertEqual(summary["output_table"], output.name)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertEqual(summary, recompute_saved_stability(
                tables, expected_molecules=1, expected_assays=1))

    def test_reaggregation_refuses_to_overwrite_different_output(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = self.write_fixture(directory)
            original = (tables / "seed_stability.csv").read_bytes()
            output = tables / "seed_stability_magnitude.csv"
            output.write_text("existing output\n")
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                recompute_saved_stability(tables, expected_molecules=1, expected_assays=1)
            self.assertEqual(output.read_text(), "existing output\n")
            self.assertEqual((tables / "seed_stability.csv").read_bytes(), original)

    def test_stability_diagnostic_keeps_negative_correlations_visible(self):
        plt = _pyplot()
        frame = pd.DataFrame({"atom_spearman_mean": [-0.9, -0.7, 0.2],
                              "bond_spearman_mean": [-0.8, -0.6, 0.3],
                              "atom_top3_overlap_mean": [0.1, 0.3, 0.5],
                              "bond_top3_overlap_mean": [0.0, 0.2, 0.4]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.png"
            with patch.object(plt, "close"):
                plot_stability(frame, path)
                figure = plt.gcf()
            try:
                self.assertEqual(len(figure.axes), 2)
                self.assertLessEqual(figure.axes[0].get_ylim()[0], -1.0)
                self.assertGreaterEqual(figure.axes[0].get_ylim()[1], 1.0)
                self.assertEqual(figure.axes[1].get_ylim()[0], 0.0)
                self.assertTrue(path.is_file())
            finally:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
