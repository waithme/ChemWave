import unittest

import numpy as np
from rdkit import Chem

from chemwave_features import chemical_relative_position_field, alternative_normalized_relative_fields
from chemwave_multitask import TargetConditionedChemWave, VARIANT_NAMES


class NormalizationSensitivityTest(unittest.TestCase):
    def test_shapes_and_expected_order(self):
        mol = Chem.MolFromSmiles("CC(=O)NCCCl")
        sqrt_field = chemical_relative_position_field(mol)
        none_field, mean_field = alternative_normalized_relative_fields(mol)
        self.assertEqual(sqrt_field.shape, (mol.GetNumAtoms(), 45))
        self.assertEqual(none_field.shape, sqrt_field.shape)
        self.assertEqual(mean_field.shape, sqrt_field.shape)
        self.assertTrue(np.all(none_field + 1e-7 >= sqrt_field))
        self.assertTrue(np.all(sqrt_field + 1e-7 >= mean_field))

    def test_parameter_count_is_controlled(self):
        counts = {
            sum(p.numel() for p in TargetConditionedChemWave(30, hidden_dim=300, variant=v).parameters())
            for v in VARIANT_NAMES
        }
        self.assertEqual(len(counts), 1)


if __name__ == "__main__":
    unittest.main()
