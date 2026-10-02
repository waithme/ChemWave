import unittest

import torch
from rdkit import Chem

from chemwave_features import chemical_relative_position_field
from chemwave_multitask import TargetConditionedChemWave, VARIANT_NAMES


class RadiusSensitivityTest(unittest.TestCase):
    def test_fixed_shape_and_parameter_count(self):
        field = chemical_relative_position_field(Chem.MolFromSmiles("CC(=O)NCCCl"))
        self.assertEqual(field.shape[1], 45)
        counts = {
            sum(p.numel() for p in TargetConditionedChemWave(30, hidden_dim=300, variant=v).parameters())
            for v in VARIANT_NAMES
        }
        self.assertEqual(len(counts), 1)

    def test_variants_are_distinct_metadata_names(self):
        self.assertEqual(VARIANT_NAMES, ("radius_1", "radius_3", "radius_5"))


if __name__ == "__main__":
    unittest.main()
