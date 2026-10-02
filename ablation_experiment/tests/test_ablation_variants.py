import unittest

import torch
from torch_geometric.data import Batch, Data

from chemwave_multitask import VARIANT_NAMES, TargetConditionedChemWave


def tiny_batch() -> Batch:
    graph = Data(
        x=torch.randn(3, 35),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 12),
        relative_position=torch.randn(3, 45),
        target_id=torch.tensor([0]),
        y=torch.tensor([1.0]),
    )
    return Batch.from_data_list([graph])


class AblationVariantTests(unittest.TestCase):
    def test_all_variants_forward_and_backward(self):
        batch = tiny_batch()
        for variant in VARIANT_NAMES:
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                model = TargetConditionedChemWave(
                    num_targets=2, hidden_dim=16, variant=variant
                )
                output = model(batch)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())

    def test_zero_initialized_additions_preserve_initial_output(self):
        batch = tiny_batch()
        outputs = {}
        for variant in VARIANT_NAMES:
            torch.manual_seed(11)
            model = TargetConditionedChemWave(
                num_targets=2, hidden_dim=16, variant=variant
            )
            model.eval()
            outputs[variant] = model(batch).detach()
        for variant, output in outputs.items():
            self.assertTrue(
                torch.equal(output, outputs["a0_v1"]),
                msg=f"Initial output differs for {variant}",
            )

    def test_target_specific_modules_match_variant_definition(self):
        expected = {
            "a0_v1": (False, False),
            "a1_relative": (True, False),
            "a2_bond": (False, True),
            "a3_full": (True, True),
            "a4_no_target_relative": (False, True),
            "a5_plain_bond_gradient": (True, True),
            "a6_no_target_bond": (True, False),
        }
        for variant, (has_relative_target, has_bond_target) in expected.items():
            model = TargetConditionedChemWave(
                num_targets=2, hidden_dim=16, variant=variant
            )
            self.assertEqual(
                hasattr(model, "target_relative_position"), has_relative_target
            )
            self.assertEqual(
                hasattr(model.blocks[0], "target_bond_conductance"),
                has_bond_target,
            )

    def test_only_a3_uses_bond_transport(self):
        for variant in VARIANT_NAMES:
            model = TargetConditionedChemWave(
                num_targets=2, hidden_dim=16, variant=variant
            )
            has_transport = hasattr(model.blocks[0], "bond_transport_scale")
            self.assertEqual(has_transport, variant == "a3_full")


if __name__ == "__main__":
    unittest.main()
