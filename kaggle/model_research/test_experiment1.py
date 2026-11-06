"""Contract checks for Experiment 1 sampling and paired scoring."""
import unittest

import numpy as np

from experiment1 import OFFSETS, choose_centers, legal_centers, metrics, quota
from analyze_experiment1 import paired_ci


class SlotSafeSamplingTests(unittest.TestCase):
    def test_full_mask_allocation_and_boundary(self):
        mask = np.ones(64, np.uint8)
        centers = choose_centers(mask, 24, train=True, seed=9)
        self.assertEqual(quota(legal_centers(mask), 24).tolist(), [6, 5, 5, 3, 5])
        self.assertEqual(centers, choose_centers(mask, 24, train=True, seed=9))
        self.assertEqual(len(set(centers)), 24)
        for c in centers:
            slot = np.searchsorted(OFFSETS, c, side="right")-1
            self.assertGreater(c, OFFSETS[slot])
            self.assertLess(c, OFFSETS[slot+1]-1)

    def test_absent_slots_and_internal_gap(self):
        mask = np.zeros(64, np.uint8)
        mask[OFFSETS[0]:OFFSETS[1]] = 1
        mask[OFFSETS[4]:OFFSETS[5]] = 1
        mask[6] = 0
        centers = choose_centers(mask, 24, train=True, seed=11)
        self.assertEqual(len(centers), 24)
        self.assertTrue(all(mask[c-1:c+2].all() for c in centers))
        self.assertTrue(all(c < OFFSETS[1] or c > OFFSETS[4] for c in centers))

    def test_k62_masks_unavailable_centers(self):
        centers = choose_centers(np.ones(64, np.uint8), 62)
        self.assertEqual(len(centers), 62)
        self.assertEqual(sum(c >= 0 for c in centers), 54)
        self.assertEqual(sum(c < 0 for c in centers), 8)

    def test_no_valid_triplet_raises(self):
        with self.assertRaises(ValueError):
            choose_centers(np.zeros(64, np.uint8), 24)


class MetricTests(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_delta(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=(40, 12)).astype(float)
        p = rng.random((40, 12))
        self.assertTrue(0 <= metrics(y, p)["macro_auc"] <= 1)
        report = paired_ci(y, p, p, draws=20)
        self.assertAlmostEqual(report["macro"]["mean_delta"], 0)


if __name__ == "__main__":
    unittest.main()
