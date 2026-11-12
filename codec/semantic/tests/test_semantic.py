"""Lightweight unit tests for the semantic codec loss, conditioning, and metrics.

Run from ``codec/semantic`` with::

    python -m unittest tests.test_semantic -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import metric
import semantic_loss
import conditioning
import evaluate
import semantic_metrics
import network
import train_options
import util
from mask_codec import code_bitrate_bytes


class TestPsnr(unittest.TestCase):
    def test_known_mse(self):
        # MSE = 100 -> PSNR = 10*log10(255^2/100) ~= 28.13 dB.
        ref = np.zeros((8, 8, 3), dtype=np.uint8)
        dist = np.full((8, 8, 3), 10, dtype=np.uint8)
        self.assertAlmostEqual(metric.psnr(ref, dist), 28.13, places=1)

    def test_uint8_high_error(self):
        # uint8 underflow would wrap to 1 -> wrong ~99 dB. float64 gives 0 dB.
        ref = np.zeros((8, 8, 3), dtype=np.uint8)
        dist = np.full((8, 8, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(metric.psnr(ref, dist), 0.0, places=4)

    def test_identical_images(self):
        ref = np.random.randint(0, 256, (16, 16, 3)).astype(np.uint8)
        self.assertEqual(metric.psnr(ref, ref), 99.99)


class TestMaskedLoss(unittest.TestCase):
    def test_masked_l1_empty_mask(self):
        x = torch.rand(2, 3, 8, 8)
        y = torch.rand(2, 3, 8, 8)
        m = torch.zeros(2, 1, 8, 8)
        self.assertAlmostEqual(float(semantic_loss.masked_l1(x, y, m)), 0.0, places=5)

    def test_region_l1_empty_fallback(self):
        S = torch.zeros(2, 1, 8, 8)
        E = torch.zeros(2, 1, 8, 8)
        x = torch.rand(2, 3, 8, 8)
        y = torch.rand(2, 3, 8, 8)
        loss, roi, bg, edge, fallback = semantic_loss.region_l1(
            x, y, S, E, 2.0, 0.5, 0.5, 'mean1')
        self.assertEqual(int(fallback), 2)
        self.assertAlmostEqual(float(loss), float((x - y).abs().mean()), places=5)

    def test_region_l1_components(self):
        S = torch.zeros(2, 1, 8, 8)
        S[:, :, :4, :] = 1.0
        E = torch.zeros(2, 1, 8, 8)
        x = torch.zeros(2, 3, 8, 8)
        y = torch.ones(2, 3, 8, 8)
        loss, roi, bg, edge, fallback = semantic_loss.region_l1(
            x, y, S, E, 2.0, 0.5, 0.5, 'mean1')
        self.assertEqual(int(fallback), 0)
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(roi), 0.0)
        self.assertGreater(float(bg), 0.0)

    def test_region_l1_mean1_is_area_normalized(self):
        # Mean-1 weighted mean must be independent of the mask support area.
        S = torch.zeros(2, 1, 8, 8)
        S[:, :, :4, :] = 1.0
        x = torch.zeros(2, 3, 8, 8)
        y = torch.ones(2, 3, 8, 8)
        # w = 1 * S + 0 * E + 0 * (1 - S) = S (all weight in the ROI).
        loss, _, _, _, _ = semantic_loss.region_l1(
            x, y, S, None, 1.0, 0.0, 0.0, 'mean1')
        # Only ROI pixels (all diff = 1) contribute; the mean1 norm removes the
        # area scaling, so the loss equals the ROI mean of |x-y| = 1.
        self.assertAlmostEqual(float(loss), 1.0, places=5)


class TestEdgeOnlyRegionLoss(unittest.TestCase):
    def test_empty_mask_with_edges_uses_region_weights(self):
        # No object pixels, one edge pixel. Disjoint weights become
        # [w_edge, w_bg] = [3, 5], sum = 8.
        S = torch.zeros(1, 1, 1, 2)
        E = torch.zeros(1, 1, 1, 2)
        E[0, 0, 0, 0] = 1.0
        x = torch.zeros(1, 3, 1, 2)
        y = torch.zeros(1, 3, 1, 2)
        y[:, :, 0, 0] = 1.0  # diff localized to the edge-only pixel.
        loss, _, _, _, fb = semantic_loss.region_l1(
            x, y, S, E, 2.0, 3.0, 5.0, 'mean1', partition='disjoint')
        self.assertEqual(int(fb), 0)
        # num = w_edge * C = 9, den = 8 * C = 24.
        self.assertAlmostEqual(float(loss), 9.0 / 24.0, places=5)

    def test_none_mask_with_edges_uses_region_path(self):
        E = torch.zeros(1, 1, 1, 2)
        E[0, 0, 0, 0] = 1.0
        x = torch.zeros(1, 3, 1, 2)
        y = torch.zeros(1, 3, 1, 2)
        y[:, :, 0, 0] = 1.0
        loss, _, _, _, fb = semantic_loss.region_l1(
            x, y, None, E, 2.0, 3.0, 5.0, 'mean1', partition='disjoint')
        self.assertEqual(int(fb), 0)
        self.assertAlmostEqual(float(loss), 9.0 / 24.0, places=5)

    def test_both_maps_empty_legacy_fallback(self):
        S = torch.zeros(2, 1, 4, 4)
        E = torch.zeros(2, 1, 4, 4)
        x = torch.rand(2, 3, 4, 4)
        y = torch.rand(2, 3, 4, 4)
        loss, _, _, _, fb = semantic_loss.region_l1(
            x, y, S, E, 2.0, 3.0, 5.0, 'mean1', partition='disjoint')
        self.assertEqual(int(fb), 2)
        self.assertAlmostEqual(float(loss), float((x - y).abs().mean()), places=5)

    def test_sum_norm_gradient_is_connected(self):
        S = torch.zeros(1, 1, 4, 4)
        S[:, :, :2, :] = 1.0
        E = torch.zeros(1, 1, 4, 4)
        E[:, :, 2:, :] = 1.0
        recon = torch.rand(1, 3, 4, 4, requires_grad=True)
        target = torch.rand(1, 3, 4, 4)
        loss, _, _, _, _ = semantic_loss.region_l1(
            recon, target, S, E, 2.0, 3.0, 5.0, 'sum', partition='disjoint')
        loss.backward()
        self.assertIsNotNone(recon.grad)
        self.assertGreater(float(recon.grad.abs().sum()), 0.0)


class TestEdgeOnlyEvaluation(unittest.TestCase):
    def test_edge_only_sample_keeps_edge_metrics(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        edge = np.zeros((8, 8), dtype=np.uint8)
        edge[3, :] = 1
        has_mask, has_bg, has_edge = evaluate._region_flags(mask, edge)
        self.assertFalse(has_mask)
        self.assertFalse(has_bg)
        self.assertTrue(has_edge)

    def test_object_sample_has_all_regions(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        edge = np.zeros((8, 8), dtype=np.uint8)
        edge[2, :] = 1
        has_mask, has_bg, has_edge = evaluate._region_flags(mask, edge)
        self.assertTrue(has_mask)
        self.assertTrue(has_bg)
        self.assertTrue(has_edge)


class TestGateDefault(unittest.TestCase):
    def test_default_gate_mode_is_none(self):
        args = train_options.parser.parse_args(['--train', 'x', '--eval', 'y'])
        self.assertEqual(args.gate_mode, 'none')


class TestGroupBoundaries(unittest.TestCase):
    def test_non_monotonic_schedule_is_sorted_by_step(self):
        schedule = [0, 3, 3, 3, 2, 2, 1, 1, 1, 1]
        self.assertEqual(
            evaluate._group_boundaries(schedule, len(schedule)),
            [1, 4, 6, 10])

    def test_object_first_schedule(self):
        schedule = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        self.assertEqual(
            evaluate._group_boundaries(schedule, len(schedule)),
            [1, 4, 6, 10])


class TestPrefixBytes(unittest.TestCase):
    def test_raw_prefix_all_ones(self):
        bits, height, width, iterations = 4, 2, 2, 3
        code = np.ones((iterations * bits, height, width), dtype=np.int8)
        out = evaluate._slot_visual_bytes(code, iterations, bits)
        per_slot = bits * height * width / 8.0
        np.testing.assert_allclose(
            out, np.arange(1, iterations + 1) * per_slot)

    def test_raw_prefix_masked_positions(self):
        bits, height, width, iterations = 2, 2, 2, 2
        code = np.ones((iterations * bits, height, width), dtype=np.int8)
        code[:bits, :, :width // 2] = 0
        out = evaluate._slot_visual_bytes(code, iterations, bits)
        first = bits * height * (width // 2) / 8.0
        second = bits * height * width / 8.0
        np.testing.assert_allclose(out, [first, first + second])

    def test_compressed_prefix_monotonic_and_final_matches_full(self):
        bits, height, width, iterations = 4, 4, 4, 3
        code = np.ones((iterations * bits, height, width), dtype=np.int8)
        out = evaluate._prefix_visual_bytes_compressed(code, iterations, bits)
        self.assertTrue(np.all(np.diff(out) >= 0))
        self.assertEqual(float(out[-1]), float(code_bitrate_bytes(code)))


class TestDisjointPartition(unittest.TestCase):
    def test_region_l1_disjoint_edge_not_object(self):
        # 1 sample, 3 channels, 2x2 grid. Pixel (0,0) is object-only,
        # (0,1) is object+edge overlap, (1,0) is edge-only, (1,1) background.
        S = torch.zeros(1, 1, 2, 2)
        E = torch.zeros(1, 1, 2, 2)
        S[0, 0, 0, 0] = 1.0
        S[0, 0, 0, 1] = 1.0
        E[0, 0, 0, 1] = 1.0
        E[0, 0, 1, 0] = 1.0
        w_roi, w_edge, w_bg = 2.0, 3.0, 5.0
        # Disjoint weights: [[2, 2], [3, 5]], sum = 12.
        x = torch.zeros(1, 3, 2, 2)
        y = torch.zeros(1, 3, 2, 2)
        y[:, :, 1, 0] = 1.0  # diff localized to the edge-only pixel.
        loss, _, _, _, fb = semantic_loss.region_l1(
            x, y, S, E, w_roi, w_edge, w_bg, 'mean1', partition='disjoint')
        # num = w_edge * 3 (channels); den = 12 * 3 -> loss = w_edge / 12.
        self.assertEqual(int(fb), 0)
        self.assertAlmostEqual(float(loss), 3.0 / 12.0, places=5)

    def test_region_l1_disjoint_background_is_w_bg(self):
        S = torch.zeros(1, 1, 1, 2)
        E = torch.zeros(1, 1, 1, 2)
        S[0, 0, 0, 0] = 1.0  # object pixel
        # disjoint weights: [w_roi, w_bg] = [2, 5], sum = 7.
        x = torch.zeros(1, 1, 1, 2)
        y = torch.zeros(1, 1, 1, 2)
        y[:, :, 0, 1] = 1.0  # diff localized to the background pixel.
        loss, _, _, _, _ = semantic_loss.region_l1(
            x, y, S, E, 2.0, 3.0, 5.0, 'mean1', partition='disjoint')
        self.assertAlmostEqual(float(loss), 5.0 / 7.0, places=5)

    def test_region_l1_disjoint_overlap_is_object(self):
        # An object+edge pixel is weighted as object only (no double-count).
        S = torch.zeros(1, 1, 1, 2)
        E = torch.zeros(1, 1, 1, 2)
        S[0, 0, 0, 0] = 1.0
        E[0, 0, 0, 0] = 1.0
        E[0, 0, 0, 1] = 1.0
        # disjoint weights: [w_roi, w_edge] = [2, 3], sum = 5.
        x = torch.zeros(1, 1, 1, 2)
        y = torch.zeros(1, 1, 1, 2)
        y[:, :, 0, 0] = 1.0  # diff localized to the overlap pixel.
        loss, _, _, _, _ = semantic_loss.region_l1(
            x, y, S, E, 2.0, 3.0, 5.0, 'mean1', partition='disjoint')
        self.assertAlmostEqual(float(loss), 2.0 / 5.0, places=5)


class TestCodeBlockMask(unittest.TestCase):
    def test_phases_zero_and_three_return_none(self):
        S = torch.ones(1, 1, 8, 8)
        E = torch.ones(1, 1, 8, 8)
        self.assertIsNone(semantic_loss.code_block_mask(0, S, E, 2, 2))
        self.assertIsNone(semantic_loss.code_block_mask(3, S, E, 2, 2))

    def test_phase1_uses_S(self):
        S = torch.zeros(1, 1, 8, 8)
        S[:, :, :, :4] = 1.0
        E = torch.zeros(1, 1, 8, 8)
        mask = semantic_loss.code_block_mask(1, S, E, 1, 2)
        self.assertTrue(torch.equal(
            mask, torch.tensor([[[[1.0, 0.0]]]])))

    def test_phase2_uses_E(self):
        S = torch.zeros(1, 1, 8, 8)
        E = torch.zeros(1, 1, 8, 8)
        E[:, :, :, :4] = 1.0
        mask = semantic_loss.code_block_mask(2, S, E, 1, 2)
        self.assertTrue(torch.equal(
            mask, torch.tensor([[[[1.0, 0.0]]]])))

    def test_threshold(self):
        S = torch.zeros(1, 1, 4, 4)
        S[0, 0, 0, 0] = 1.0  # 1/16 coverage = 0.0625
        dropped = semantic_loss.code_block_mask(1, S, None, 1, 1, threshold=0.5)
        self.assertTrue(torch.equal(dropped, torch.zeros(1, 1, 1, 1)))
        kept = semantic_loss.code_block_mask(1, S, None, 1, 1)
        self.assertTrue(torch.equal(kept, torch.ones(1, 1, 1, 1)))


class TestPhaseSchedule(unittest.TestCase):
    def test_phase_of_step(self):
        schedule = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]
        self.assertEqual(semantic_loss.phase_of_step(0, schedule), 0)
        self.assertEqual(semantic_loss.phase_of_step(1, schedule), 1)
        self.assertEqual(semantic_loss.phase_of_step(4, schedule), 2)
        self.assertEqual(semantic_loss.phase_of_step(6, schedule), 3)
        self.assertEqual(semantic_loss.phase_of_step(99, schedule), 3)

    def test_phase_of_step_string(self):
        self.assertEqual(semantic_loss.phase_of_step(2, '0,1,1,1,2,2,3,3,3,3'), 1)

    def test_phase_multipliers(self):
        w_roi = [1.0, 4.0, 1.0, 0.5]
        w_edge = [1.0, 0.25, 2.0, 0.5]
        w_bg = [1.0, 0.25, 0.5, 1.0]
        roi, edge, bg = semantic_loss.phase_multipliers(1, w_roi, w_edge, w_bg)
        self.assertAlmostEqual(roi, 4.0)
        self.assertAlmostEqual(edge, 0.25)
        self.assertAlmostEqual(bg, 0.25)

    def test_gate_for_phase(self):
        S = torch.ones(2, 1, 8, 8)
        E = torch.zeros(2, 1, 8, 8)
        self.assertIsNone(semantic_loss.gate_for_phase(0, None, None, 'mask'))
        self.assertIsNone(semantic_loss.gate_for_phase(1, None, None, 'mask'))
        self.assertTrue(torch.equal(
            semantic_loss.gate_for_phase(1, S, E, 'mask'), S))
        self.assertTrue(torch.equal(
            semantic_loss.gate_for_phase(2, S, E, 'mask'), E))
        full = semantic_loss.gate_for_phase(0, S, E, 'mask')
        self.assertTrue(torch.all(full == 1.0))
        self.assertIsNone(semantic_loss.gate_for_phase(1, S, E, 'none'))


class TestFeatheredMask(unittest.TestCase):
    def test_hard_gate(self):
        S = torch.zeros(1, 1, 8, 8)
        S[:, :, 2:6, 2:6] = 1.0
        S_f, E_f = util.feathered_mask(S, None, 0)
        self.assertTrue(torch.equal(S_f, S))
        self.assertIsNone(E_f)

    def test_feather_softens(self):
        S = torch.zeros(1, 1, 8, 8)
        S[:, :, 3:5, 3:5] = 1.0
        S_f, _ = util.feathered_mask(S, None, 1)
        self.assertEqual(S_f.shape, S.shape)
        # The feathered mask must have intermediate values at the boundary.
        self.assertTrue(((S_f > 0.0) & (S_f < 1.0)).any())
        # And it must preserve the interior.
        self.assertAlmostEqual(float(S_f[:, :, 3:5, 3:5].max()), 1.0, places=5)


class TestSpatialFiLM(unittest.TestCase):
    def test_zero_init_identity(self):
        film = conditioning.SpatialFiLM(2, 8, init='zero')
        x = torch.rand(2, 8, 16, 16)
        sem = torch.rand(2, 2, 64, 64)
        self.assertTrue(torch.allclose(film(x, sem), x, atol=1e-6))

    def test_random_init_not_identity(self):
        film = conditioning.SpatialFiLM(2, 8, init='random')
        x = torch.rand(2, 8, 16, 16)
        sem = torch.rand(2, 2, 64, 64)
        self.assertFalse(torch.allclose(film(x, sem), x, atol=1e-3))

    def test_mask_add_zero_init(self):
        add = conditioning.MaskAdd(2, 8, init='zero')
        sem = torch.rand(2, 2, 64, 64)
        out = add(sem, 16, 16)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))


class TestEarlyExit(unittest.TestCase):
    def test_output_shape_and_range(self):
        head = network.EarlyExit(shuffle_up=4, conv_in=16)
        x = torch.rand(2, 16, 4, 4)
        out = head(x)
        self.assertEqual(tuple(out.shape), (2, 3, 16, 16))
        self.assertTrue(float(out.min()) >= -0.5 - 1e-5)
        self.assertTrue(float(out.max()) <= 0.5 + 1e-5)

    def test_head_has_capacity(self):
        head = network.EarlyExit(shuffle_up=16, conv_in=512)
        x = torch.rand(1, 512, 2, 2)
        out = head(x)
        self.assertEqual(tuple(out.shape), (1, 3, 32, 32))

    def test_src_head_keeps_reference_keys(self):
        head = network.EarlyExit(shuffle_up=4, conv_in=16, ee_head='src')
        self.assertEqual(set(head.state_dict().keys()), {'conv.weight'})

    def test_refined_is_identity_at_init(self):
        src = network.EarlyExit(shuffle_up=4, conv_in=16, ee_head='src')
        refined = network.EarlyExit(
            shuffle_up=4, conv_in=16, ee_head='refined')
        refined.conv.load_state_dict(src.conv.state_dict())
        x = torch.rand(2, 16, 4, 4)
        self.assertTrue(torch.allclose(src(x), refined(x), atol=1e-6))
        self.assertEqual(float(refined.extra_conv.weight.abs().sum()), 0.0)

    def test_refined_extra_branch_gets_gradient(self):
        refined = network.EarlyExit(
            shuffle_up=4, conv_in=16, ee_head='refined')
        x = torch.rand(1, 16, 4, 4)
        target = torch.rand(1, 3, 16, 16) - 0.5
        loss = (refined(x) - target).abs().mean()
        loss.backward()
        self.assertIsNotNone(refined.extra_conv.weight.grad)
        self.assertGreater(
            float(refined.extra_conv.weight.grad.abs().sum()), 0.0)


class TestRoiMetrics(unittest.TestCase):
    def test_bbox_from_mask(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        self.assertEqual(semantic_metrics.bbox_from_mask(mask), (2, 4, 3, 6))

    def test_bbox_from_empty_mask(self):
        self.assertIsNone(semantic_metrics.bbox_from_mask(np.zeros((8, 8))))

    def test_psnr_masked_known(self):
        ref = np.zeros((8, 8, 3), dtype=np.uint8)
        dist = np.full((8, 8, 3), 10, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8)
        self.assertAlmostEqual(semantic_metrics.psnr_masked(ref, dist, mask), 28.13, places=1)

    def test_psnr_masked_empty(self):
        ref = np.zeros((8, 8, 3), dtype=np.uint8)
        dist = np.full((8, 8, 3), 10, dtype=np.uint8)
        self.assertEqual(semantic_metrics.psnr_masked(ref, dist, np.zeros((8, 8))), 99.99)


@unittest.skipUnless(
    __import__('importlib').util.find_spec('lpips') is not None,
    'official lpips package not installed')
class TestOfficialLpips(unittest.TestCase):
    def test_weighted_lpips_scalar(self):
        x = torch.rand(2, 3, 32, 32) - 0.5
        y = torch.rand(2, 3, 32, 32) - 0.5
        W = torch.ones(2, 1, 32, 32)
        out = semantic_loss.weighted_lpips(x, y, W, weight=0.5, net='alex')
        self.assertEqual(out.dim(), 0)
        self.assertGreaterEqual(float(out), 0.0)

    def test_weighted_lpips_zero_on_identical(self):
        x = torch.rand(1, 3, 32, 32) - 0.5
        out = semantic_loss.weighted_lpips(x, x, None, weight=1.0, net='alex')
        self.assertLess(float(out), 1e-3)

    def test_batched_map_matches_single(self):
        ref = torch.rand(3, 32, 32)
        dists = [torch.rand(3, 32, 32), torch.rand(3, 32, 32)]
        batched = semantic_metrics.lpips_map_batch(ref, dists, net='alex')
        single = semantic_metrics.lpips_map(ref, dists[0], net='alex')
        self.assertEqual(tuple(batched.shape), (2, 1, 32, 32))
        self.assertTrue(torch.allclose(batched[0:1], single, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
