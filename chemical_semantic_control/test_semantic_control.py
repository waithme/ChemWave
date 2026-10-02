import unittest

import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch

from chemwave_features import (
    chemical_relative_position_field,
    chemical_roles,
    element_channels,
    fixed_random_roles,
    molecule_to_graph35,
)
from chemwave_multitask import TargetConditionedChemWave, VARIANT_NAMES


class SemanticControlTest(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.MolFromSmiles("CCOc1ncc(S(=O)(=O)NCCCl)cc1Br")

    def test_element_channels_are_exclusive_and_nine_dimensional(self):
        channels = element_channels(self.mol)
        self.assertEqual(channels.shape, (self.mol.GetNumAtoms(), 9))
        np.testing.assert_array_equal(channels.sum(axis=1), 1.0)

    def test_fixed_random_roles_preserve_marginals_and_are_deterministic(self):
        roles = chemical_roles(self.mol)
        first = fixed_random_roles(self.mol, roles)
        second = fixed_random_roles(self.mol, roles)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=0), roles.sum(axis=0))
        self.assertEqual(sorted(map(tuple, first)), sorted(map(tuple, roles)))

    def test_graph_fields_and_forward(self):
        graph = molecule_to_graph35("CCOc1ncc(S(=O)(=O)NCCCl)cc1Br", 7.0)
        self.assertEqual(tuple(graph.relative_position.shape)[1], 45)
        self.assertEqual(tuple(graph.relative_position_element.shape)[1], 45)
        self.assertEqual(tuple(graph.relative_position_random.shape)[1], 45)
        self.assertNotIn("cliff_label", graph.keys())
        batch = Batch.from_data_list([graph])
        for variant in VARIANT_NAMES:
            output = TargetConditionedChemWave(
                30, hidden_dim=300, variant=variant
            )(batch)
            self.assertEqual(tuple(output.shape), (1,))

    def test_parameter_count_is_identical(self):
        counts = {
            sum(
                p.numel()
                for p in TargetConditionedChemWave(
                    30, hidden_dim=300, variant=variant
                ).parameters()
            )
            for variant in VARIANT_NAMES
        }
        self.assertEqual(counts, {2_629_896})

    def test_variant_names(self):
        self.assertEqual(VARIANT_NAMES, ("element_shell", "random_role"))


if __name__ == "__main__":
    unittest.main()
