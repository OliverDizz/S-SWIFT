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
import semantic_metrics


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
            x, y, S, E, S, 2.0, 0.5, 0.5)
        self.assertEqual(int(fallback), 2)
        self.assertAlmostEqual(float(loss), float((x - y).abs().mean()), places=5)

    def test_region_l1_components(self):
        S = torch.zeros(2, 1, 8, 8)
        S[:, :, :4, :] = 1.0
        E = torch.zeros(2, 1, 8, 8)
        x = torch.zeros(2, 3, 8, 8)
        y = torch.ones(2, 3, 8, 8)
        loss, roi, bg, edge, fallback = semantic_loss.region_l1(
            x, y, S, E, S, 2.0, 0.5, 0.5)
        self.assertEqual(int(fallback), 0)
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(roi), 0.0)
        self.assertGreater(float(bg), 0.0)

    def test_build_roi_map_no_band(self):
        S = torch.ones(1, 1, 8, 8)
        self.assertTrue(torch.equal(semantic_loss.build_roi_map(S, 0, 0.5), S))

    def test_ramp_multipliers_linear(self):
        roi_start, bg_start = semantic_loss.ramp_multipliers(0, 10, 'linear', 0.5)
        roi_end, bg_end = semantic_loss.ramp_multipliers(9, 10, 'linear', 0.5)
        self.assertAlmostEqual(roi_start, 0.5, places=5)
        self.assertAlmostEqual(roi_end, 1.0, places=5)
        self.assertGreater(bg_start, bg_end)


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


if __name__ == '__main__':
    unittest.main()
