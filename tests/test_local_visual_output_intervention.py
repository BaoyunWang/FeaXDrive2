# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""CPU tests of full local-output ablation; no datasets/checkpoints required.

python -m pytest tests/test_local_visual_output_intervention.py
"""

import json
import unittest

import torch

from feaxdrive2.model.local_visual_output_intervention import LocalVisualOutputIntervention
from .helpers import make_branch


class LocalVisualOutputInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.branch, self.inputs = make_branch()

    def calls(self, count=5):
        return [self.branch(**dict(self.inputs, flow_time=torch.tensor([k / 5.])))
                for k in range(count)]

    def test_normal_bitwise_no_rng_no_input_or_state_changes(self):
        tensors = {k: v.clone() for k, v in self.branch.state_dict().items()}
        inputs = {k: v.clone() for k, v in self.inputs.items()}
        observed = []
        observer = self.branch.visual_attention.register_forward_hook(
            lambda module, args, output: observed.append(output[0].clone()))
        try:
            with torch.no_grad():
                expected = self.calls()
                observed.clear()
                rng = torch.get_rng_state().clone()
                with LocalVisualOutputIntervention(self.branch) as hook:
                    actual = self.calls()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        finally:
            observer.remove()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertEqual([r["flow_index"] for r in hook.steps], list(range(5)))
        for row, raw in zip(hook.steps, observed):
            self.assertAlmostEqual(row["input_output_rms"], float(raw.double().square().mean().sqrt()))
            self.assertAlmostEqual(row["input_output_mean_norm"], float(raw.double().norm(dim=-1).mean()))
            self.assertEqual(row["delivered_output_rms"], row["input_output_rms"])
            self.assertEqual(row["delivered_output_mean_norm"], row["input_output_mean_norm"])
        json.dumps(hook.steps, allow_nan=False)
        self.assertTrue(all(torch.equal(tensors[k], v) for k, v in self.branch.state_dict().items()))
        self.assertTrue(all(torch.equal(inputs[k], v) for k, v in self.inputs.items()))
        self.assertFalse(self.branch.visual_attention._forward_hooks)

    def test_disabled_removes_bias_and_null_but_runs_scene_attention(self):
        # The intervention removes a nonzero output even when no physical key is valid.
        for invalid in (False, True):
            with self.subTest(all_invalid=invalid), torch.no_grad():
                self.inputs["camera_valid"].fill_(not invalid)
                self.branch.visual_attention.out_proj.weight.zero_()
                self.branch.visual_attention.out_proj.bias.fill_(.75)
                state = {k: v.clone() for k, v in self.branch.state_dict().items()}
                expected = self.calls()
                outputs, scenes = [], []
                scene_hook = self.branch.scene_attention.register_forward_hook(
                    lambda module, args, output: scenes.append(output[0].clone()))
                try:
                    rng = torch.get_rng_state().clone()
                    with LocalVisualOutputIntervention(self.branch, disabled=True) as hook:
                        observer = self.branch.visual_attention.register_forward_hook(
                            lambda module, args, output: outputs.append(output[0].clone()))
                        try:
                            actual = self.calls()
                        finally:
                            observer.remove()
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                finally:
                    scene_hook.remove()
                self.assertEqual(len(outputs), 5)
                self.assertEqual(len(scenes), 5)
                self.assertTrue(all(torch.count_nonzero(v) == 0 for v in outputs))
                self.assertTrue(all(torch.isfinite(v).all() for v in actual))
                self.assertTrue(any(torch.count_nonzero(v) > 0 for v in scenes))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(expected, actual)))
                for row in hook.steps:
                    self.assertEqual(row["input_output_rms"], .75)
                    self.assertEqual(row["input_output_mean_norm"], 3.)
                    self.assertEqual(row["delivered_output_rms"], 0.)
                    self.assertEqual(row["delivered_output_mean_norm"], 0.)
                if invalid:
                    self.assertFalse(self.branch.last_projection_valid.any())
                self.assertTrue(all(torch.equal(state[k], v) for k, v in self.branch.state_dict().items()))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, self.calls())))

    def test_tuple_identity_normal_and_weight_identity_disabled(self):
        query = torch.zeros(1, 1, self.branch.width)
        for disabled in (False, True):
            with self.subTest(disabled=disabled), torch.no_grad():
                seen = []
                observer = self.branch.visual_attention.register_forward_hook(
                    lambda module, args, output: seen.append(output))
                try:
                    with LocalVisualOutputIntervention(self.branch, disabled=disabled):
                        for _ in range(5):
                            result = self.branch.visual_attention(query, query, query, need_weights=True)
                            self.assertIs(result[1], seen[-1][1])
                            if disabled:
                                self.assertEqual(torch.count_nonzero(result[0]), 0)
                                self.assertIsNot(result[0], seen[-1][0])
                            else:
                                self.assertIs(result, seen[-1])
                finally:
                    observer.remove()

    def test_grad_eval_and_frozen_guards(self):
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            with LocalVisualOutputIntervention(self.branch):
                pass
        with torch.no_grad():
            self.branch.visual_attention.train()
            with self.assertRaisesRegex(RuntimeError, "eval"):
                with LocalVisualOutputIntervention(self.branch):
                    pass
            self.branch.eval()
            next(self.branch.parameters()).requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, "frozen"):
                with LocalVisualOutputIntervention(self.branch):
                    pass
        self.assertFalse(self.branch.visual_attention._forward_hooks)

    def test_guard_is_checked_during_execution(self):
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "no_grad"):
            with LocalVisualOutputIntervention(self.branch):
                with torch.enable_grad():
                    self.calls(1)
        self.assertFalse(self.branch.visual_attention._forward_hooks)

    def test_exactly_five_calls_and_cleanup(self):
        for count in (0, 4, 6):
            with self.subTest(count=count), torch.no_grad():
                with self.assertRaisesRegex(RuntimeError, "exactly five"):
                    with LocalVisualOutputIntervention(self.branch, disabled=True):
                        self.calls(count)
                self.assertFalse(self.branch.visual_attention._forward_hooks)
                with LocalVisualOutputIntervention(self.branch):
                    self.calls()

    def test_exception_keeps_original_error_and_allows_reuse(self):
        hook = LocalVisualOutputIntervention(self.branch, disabled=True)
        with torch.no_grad():
            with self.assertRaisesRegex(LookupError, "original"):
                with hook:
                    self.calls(1)
                    raise LookupError("original")
            self.assertFalse(self.branch.visual_attention._forward_hooks)
            with hook:
                self.calls()
            self.assertEqual(len(hook.steps), 5)

    def test_nested_context_rejected_without_removing_outer_hook(self):
        with torch.no_grad(), LocalVisualOutputIntervention(self.branch) as outer:
            for nested in (outer, LocalVisualOutputIntervention(self.branch, disabled=True)):
                with self.assertRaisesRegex(RuntimeError, "already has"):
                    with nested:
                        pass
                self.assertEqual(len(self.branch.visual_attention._forward_hooks), 1)
            self.calls()
        self.assertFalse(self.branch.visual_attention._forward_hooks)

    def test_nonfinite_output_fails_and_cleans_up(self):
        with torch.no_grad():
            self.branch.visual_attention.out_proj.bias.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                with LocalVisualOutputIntervention(self.branch, disabled=True):
                    self.calls(1)
        self.assertFalse(self.branch.visual_attention._forward_hooks)


if __name__ == "__main__":
    unittest.main()
