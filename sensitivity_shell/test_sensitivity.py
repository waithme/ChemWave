import unittest

import numpy as np
from rdkit import Chem

from chemwave_features import chemical_relative_position_field, cumulative_relative_position_field
from chemwave_multitask import TargetConditionedChemWave, VARIANT_NAMES


class ShellSensitivityTest(unittest.TestCase):
    def test_shapes_and_definitions(self):
        mol = Chem.MolFromSmiles("CC(=O)NCCCl")
        exact = chemical_relative_position_field(mol)
        cumulative = cumulative_relative_position_field(mol)
        self.assertEqual(exact.shape, cumulative.shape)
        self.assertEqual(exact.shape[1], 45)
        np.testing.assert_allclose(exact[:, :9], cumulative[:, :9])
        self.assertFalse(np.allclose(exact, cumulative))

    def test_parameter_count_is_controlled(self):
        counts = {
            sum(p.numel() for p in TargetConditionedChemWave(30, hidden_dim=300, variant=v).parameters())
            for v in VARIANT_NAMES
        }
        self.assertEqual(len(counts), 1)


if __name__ == "__main__":
    unittest.main()
