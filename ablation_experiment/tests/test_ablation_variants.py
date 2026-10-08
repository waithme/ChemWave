import unittest

import torch
from torch_geometric.data import Batch, Data

from chemwave_multitask import (
    VARIANT_NAMES, VARIANT_ALIASES, TargetConditionedChemWave,
    canonical_variant_name,
)
from chemwave_training import validate_checkpoint
from run_chemwave_finetune_test import result_key


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
            "a3_affine_transport": (True, True),
            "a4_no_target_relative": (False, True),
            "a5_full": (True, True),
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
            self.assertEqual(has_transport, variant == "a3_affine_transport")

    def test_default_is_final_full_without_affine_transport(self):
        model = TargetConditionedChemWave(num_targets=2, hidden_dim=16)
        self.assertEqual(model.variant, "a5_full")
        self.assertFalse(model.config["use_bond_transport"])

    def test_legacy_aliases_preserve_weights_and_predictions(self):
        batch = tiny_batch()
        for legacy, canonical in VARIANT_ALIASES.items():
            with self.subTest(legacy=legacy):
                torch.manual_seed(19)
                old = TargetConditionedChemWave(2, hidden_dim=16, variant=legacy)
                torch.manual_seed(19)
                new = TargetConditionedChemWave(2, hidden_dim=16, variant=canonical)
                self.assertEqual(old.variant, canonical)
                self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
                for key, value in old.state_dict().items():
                    self.assertTrue(torch.equal(value, new.state_dict()[key]))
                new.load_state_dict(old.state_dict(), strict=True)
                old.eval(); new.eval()
                self.assertTrue(torch.equal(old(batch), new(batch)))
        self.assertEqual(len(VARIANT_NAMES), 7)
        self.assertNotIn("a3_full", VARIANT_NAMES)
        self.assertNotIn("a5_plain_bond_gradient", VARIANT_NAMES)

    def test_checkpoint_aliases_are_equivalent_but_architectures_are_not(self):
        data = dict(development_data_sha256="test-fingerprint", target_names=["A", "B"])
        expected = dict(stage="shared_pretrain", seed=0, data=data)
        for legacy, canonical in VARIANT_ALIASES.items():
            checkpoint = dict(stage="shared_pretrain", seed=0, model_state={},
                              variant=legacy, **data)
            validate_checkpoint(checkpoint, variant=canonical, **expected)
            checkpoint["variant"] = canonical
            validate_checkpoint(checkpoint, variant=legacy, **expected)
        checkpoint = dict(stage="shared_pretrain", seed=0, model_state={},
                          variant="a3_full", **data)
        with self.assertRaises(RuntimeError):
            validate_checkpoint(checkpoint, variant="a5_full", **expected)
        for field, bad in (("variant", None), ("variant", "unknown"),
                           ("seed", 1), ("development_data_sha256", "wrong")):
            checkpoint = dict(stage="shared_pretrain", seed=0, model_state={},
                              variant="a5_plain_bond_gradient", **data)
            checkpoint[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(RuntimeError):
                validate_checkpoint(checkpoint, variant="a5_full", **expected)

    def test_result_keys_deduplicate_aliases_not_different_architectures(self):
        row = dict(variant="a5_plain_bond_gradient", development_data_sha256="fp",
                   target="A", seed=0)
        canonical = dict(row, variant="a5_full")
        affine = dict(row, variant="a3_affine_transport")
        self.assertEqual(result_key(row), result_key(canonical))
        self.assertNotEqual(result_key(row), result_key(affine))
        with self.assertRaises(ValueError):
            canonical_variant_name("unknown")


if __name__ == "__main__":
    unittest.main()
