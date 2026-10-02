from __future__ import annotations

import unittest

import numpy as np
from rdkit import Chem

from chemwave_features import CHEMICAL_ROLE_NAMES, chemical_roles


def positive_atom_indices(smiles: str) -> set[int]:
    return role_atom_indices(smiles, "positive_ionizable")


def role_atom_indices(smiles: str, role_name: str) -> set[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid test SMILES: {smiles}")
    roles = chemical_roles(mol)
    role_index = CHEMICAL_ROLE_NAMES.index(role_name)
    return {
        atom_index
        for atom_index, value in enumerate(roles[:, role_index])
        if value == 1.0
    }


class PositiveIonizableRoleTests(unittest.TestCase):
    def test_primary_secondary_and_tertiary_amines_are_positive(self):
        for smiles in ("CN", "CNC", "CN(C)C"):
            mol = Chem.MolFromSmiles(smiles)
            nitrogen = next(
                atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7
            )
            self.assertIn(nitrogen, positive_atom_indices(smiles))

    def test_quaternary_ammonium_is_positive(self):
        self.assertEqual(positive_atom_indices("C[N+](C)(C)C"), {1})

    def test_nitrile_amide_aniline_and_nitro_are_not_positive(self):
        for smiles in (
            "CC#N",
            "CC(=O)N",
            "c1ccccc1N",
            "[O-][N+](=O)c1ccccc1",
        ):
            self.assertEqual(positive_atom_indices(smiles), set())

    def test_guanidine_and_imidazole_groups_are_positive(self):
        self.assertEqual(len(positive_atom_indices("NC(=N)N")), 4)
        self.assertEqual(len(positive_atom_indices("c1ncc[nH]1")), 5)


class CompleteChemicalRoleTests(unittest.TestCase):
    def test_donor_and_acceptor_roles(self):
        self.assertEqual(role_atom_indices("CCO", "donor"), {2})
        self.assertEqual(role_atom_indices("CCO", "acceptor"), {2})

        # In acetamide, the amide nitrogen is a donor but not an acceptor;
        # the carbonyl oxygen is the acceptor.
        self.assertEqual(role_atom_indices("CC(=O)N", "donor"), {3})
        self.assertEqual(role_atom_indices("CC(=O)N", "acceptor"), {2})

    def test_negative_ionizable_role_for_acid_and_anion(self):
        self.assertEqual(
            role_atom_indices("CC(=O)O", "negative_ionizable"), {3}
        )
        self.assertEqual(
            role_atom_indices("CC(=O)[O-]", "negative_ionizable"), {3}
        )

    def test_aromatic_and_hydrophobe_roles(self):
        benzene_atoms = set(range(6))
        self.assertEqual(
            role_atom_indices("c1ccccc1", "aromatic"), benzene_atoms
        )
        self.assertEqual(
            role_atom_indices("c1ccccc1", "hydrophobe"), benzene_atoms
        )
        self.assertEqual(role_atom_indices("CCC", "hydrophobe"), {0, 1, 2})

    def test_halogen_role(self):
        self.assertEqual(role_atom_indices("Clc1ccccc1", "halogen"), {0})

    def test_ring_junction_role(self):
        self.assertEqual(
            role_atom_indices("c1ccc2ccccc2c1", "ring_junction"), {3, 8}
        )

    def test_rotatable_connector_role(self):
        self.assertEqual(
            role_atom_indices("CCCC", "rotatable_connector"), {1, 2}
        )

    def test_roles_are_nonexclusive(self):
        # The phenolic oxygen is intentionally both a donor and an acceptor.
        self.assertEqual(role_atom_indices("Oc1ccccc1", "donor"), {0})
        self.assertEqual(role_atom_indices("Oc1ccccc1", "acceptor"), {0})

    def test_output_is_binary_deterministic_and_has_nine_columns(self):
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1Cl")
        first = chemical_roles(mol)
        second = chemical_roles(mol)

        self.assertEqual(first.shape, (mol.GetNumAtoms(), 9))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.all((first == 0.0) | (first == 1.0)))
        np.testing.assert_array_equal(first, second)

    def test_roles_follow_the_supplied_graph_state(self):
        # These assertions document graph-state dependence; they do not claim
        # which protonation state predominates under an experimental condition.
        self.assertEqual(role_atom_indices("n1ccccc1", "acceptor"), {0})
        self.assertEqual(
            role_atom_indices("[nH+]1ccccc1", "acceptor"), set()
        )
        self.assertEqual(
            role_atom_indices("[nH+]1ccccc1", "positive_ionizable"), {0}
        )


if __name__ == "__main__":
    unittest.main()
