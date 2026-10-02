from __future__ import annotations

import unittest

from rdkit import Chem

from chemwave_features import CHEMICAL_ROLE_NAMES, chemical_roles


POSITIVE_ROLE = CHEMICAL_ROLE_NAMES.index("positive_ionizable")


def positive_atom_indices(smiles: str) -> set[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid test SMILES: {smiles}")
    roles = chemical_roles(mol)
    return {
        atom_index
        for atom_index, value in enumerate(roles[:, POSITIVE_ROLE])
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


if __name__ == "__main__":
    unittest.main()
