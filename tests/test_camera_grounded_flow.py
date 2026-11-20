# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Focused tests of the actual causal-prefix helper; no model/data substitute."""

import unittest
import torch

from feaxdrive2.model.camera_grounded_flow import discrete_flow_time, rollout_prefix


class CameraFlowTests(unittest.TestCase):
    def test_prefix_restores_amp_cache_on_normal_and_exceptional_exit(self):
        original_cache = torch.is_autocast_cache_enabled()
        try:
            for outer_cache in (True, False):
                torch.set_autocast_cache_enabled(outer_cache)
                for fail in (False, True):
                    def velocity(x, t, previous):
                        self.assertFalse(torch.is_autocast_cache_enabled())
                        self.assertFalse(torch.is_grad_enabled())
                        self.assertTrue(torch.is_autocast_cpu_enabled())
                        self.assertEqual(torch.get_autocast_cpu_dtype(), torch.bfloat16)
                        if fail:
                            raise RuntimeError("prefix failure")
                        return torch.ones_like(x)
                    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=outer_cache):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "prefix failure"):
                                rollout_prefix(torch.zeros(1,8,3), torch.tensor([.2]), velocity, lambda x:x)
                        else:
                            rollout_prefix(torch.zeros(1,8,3), torch.tensor([.2]), velocity, lambda x:x)
                        self.assertEqual(torch.is_autocast_cache_enabled(), outer_cache)
        finally:
            torch.set_autocast_cache_enabled(original_cache)

    def test_metric_origin_at_step_zero_is_not_denormalized_latent_zero(self):
        calls = []
        def velocity(*args):
            calls.append(args)
            return torch.ones_like(args[0])
        result = rollout_prefix(torch.zeros(2, 8, 3), torch.zeros(2), velocity, lambda x: x + 31.8)
        self.assertFalse(calls)
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_prefix_matches_manual_rollout_and_uses_pre_euler_clean_estimate(self):
        noise = torch.full((5, 8, 3), .1, requires_grad=True)
        original = noise.detach().clone()
        times = torch.arange(5).float() / 5
        seen = []
        def velocity(x, t, previous):
            self.assertFalse(torch.is_grad_enabled())
            seen.append((x.clone(), previous.clone()))
            return x * .2 + previous * .01 + .1
        result = rollout_prefix(noise, times, velocity, lambda x: 2*x + 3)
        x = original.clone()
        previous = torch.zeros_like(x)
        expected = torch.zeros_like(x)
        for step in range(4):
            v = x*.2 + previous*.01 + .1
            previous = 2*(x + (1-step/5)*v).clamp(-1, 1) + 3
            x = x + .2*v
            expected[step+1] = previous[step+1]
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(noise.detach(), original, rtol=0, atol=0)
        self.assertFalse(result.requires_grad)
        self.assertEqual(len(seen), 4)

    def test_only_query_is_clipped_and_invalid_time_rejected(self):
        seen = []
        def velocity(x, t, previous):
            seen.append(x.clone())
            return torch.ones_like(x)*20
        result = rollout_prefix(torch.ones(1,8,3)*2, torch.tensor([.8]), velocity, lambda x:x)
        self.assertGreater(seen[-1].max().item(), 1)
        self.assertEqual(result.max().item(), 1)
        for t in (torch.tensor([1.]), torch.tensor([-.1]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                discrete_flow_time(t)
        torch.testing.assert_close(discrete_flow_time(torch.tensor([.1,.3,.5,.7,.9])),
                                   torch.tensor([0.,.2,.4,.6,.8]))


if __name__ == "__main__":
    unittest.main()
