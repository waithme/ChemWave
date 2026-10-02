from __future__ import annotations

import unittest

import pandas as pd

from summarize_size_adjusted_interventions import (
    add_size_adjusted_metrics,
    collapse_seeds,
    target_contrasts,
)


class SizeAdjustedInterventionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {
                    "target": "T1",
                    "row_id": row_id,
                    "smiles": smiles,
                    "cliff": cliff,
                    "seed": seed,
                    "relative_total_abs": relative,
                    "bond_total_abs": bond,
                }
                for row_id, smiles, cliff, relative, bond in (
                    (1, "CC", 0, 2.0, 0.4),
                    (2, "CCCC", 1, 8.0, 0.8),
                )
                for seed in (0, 1)
            ]
        )

    def test_normalizes_by_rdkit_heavy_atom_count(self):
        adjusted = add_size_adjusted_metrics(self.frame)
        ethane = adjusted.loc[adjusted["row_id"] == 1].iloc[0]
        butane = adjusted.loc[adjusted["row_id"] == 2].iloc[0]
        self.assertEqual(ethane["heavy_atom_count"], 2)
        self.assertEqual(butane["heavy_atom_count"], 4)
        self.assertAlmostEqual(ethane["relative_mean_abs_per_heavy_atom"], 1.0)
        self.assertAlmostEqual(butane["relative_mean_abs_per_heavy_atom"], 2.0)

    def test_collapses_seeds_before_target_contrast(self):
        molecules = collapse_seeds(add_size_adjusted_metrics(self.frame))
        self.assertEqual(len(molecules), 2)
        contrasts = target_contrasts(molecules)
        self.assertEqual(len(contrasts), 1)
        self.assertAlmostEqual(
            contrasts.loc[
                0, "relative_mean_abs_per_heavy_atom_cliff_minus_noncliff"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
