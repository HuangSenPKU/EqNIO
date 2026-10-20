"""Verify initial-only alignment, including future-reference independence."""
import unittest
import numpy as np
from evaluate_initial_alignment import (
    initial_transform,
    cumulative_distance_index, cumulative_distance_transform,
)


class InitialAlignmentTests(unittest.TestCase):
    def test_cumulative_distance_uses_path_length_and_shared_interval(self):
        ref = np.array([[0., 0.], [6., 0.], [6., 4.], [50., -30.]])
        # A scaled, rotated prediction reaches 10 metres at a different time.
        expected = np.array([[0., -1.], [1., 0.]])
        pred = 2 * ref @ expected + [3., 8.]
        j = cumulative_distance_index(ref, 10.)
        self.assertEqual(j, 2)
        self.assertLess(np.linalg.norm(ref[j] - ref[0]), 10.)
        r, tr, angle = cumulative_distance_transform(pred, ref, 10.)
        np.testing.assert_allclose(r, expected, atol=1e-12)
        np.testing.assert_allclose(pred[0] @ r.T + tr, ref[0], atol=1e-12)
        r2, tr2, angle2 = cumulative_distance_transform(pred[:j + 1], ref[:j + 1])
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_array_equal(tr, tr2)
        self.assertEqual(angle, angle2)

    def test_cumulative_distance_stops_at_first_crossing(self):
        ref = np.array([[0., 0.], [4., 0.], [8., 0.], [12., 0.], [12., 8.]])
        self.assertEqual(cumulative_distance_index(ref), 3)
        with self.assertRaisesRegex(ValueError, "never reaches"):
            cumulative_distance_index(ref[:3])

    def test_zero_accumulated_prediction_displacement_rejects_heading(self):
        pred = np.array([[0., 0.], [1., 0.], [0., 0.], [10., 0.]])
        ref = np.array([[5., 4.], [5., 9.], [5., 14.], [5., 24.]])
        with self.assertRaisesRegex(ValueError, "zero"):
            cumulative_distance_transform(pred, ref)

    def test_initial_transform_and_future_independence(self):
        t = np.arange(181.)
        pred = np.column_stack([np.maximum(t - 20, 0), .1 * np.maximum(t - 20, 0)])
        angle = .6
        expected = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        ref = pred @ expected.T + [45, -30]
        i, j = 0, cumulative_distance_index(ref)
        r, tr, _ = initial_transform(pred, ref, i, j)
        np.testing.assert_allclose(r, expected, atol=1e-12)
        np.testing.assert_allclose(pred @ r.T + tr, ref, atol=1e-12)
        ref[j + 1:] += np.column_stack([t[j + 1:] ** 2, -t[j + 1:]])
        self.assertEqual(cumulative_distance_index(ref), j)
        r2, tr2, _ = initial_transform(pred, ref, i, j)
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_array_equal(tr, tr2)

    def test_does_not_fit_scale_or_hide_late_drift(self):
        pred = np.column_stack([np.arange(181.), np.zeros(181)])
        ref = pred * 2 + [10, 20]
        r, tr, _ = initial_transform(pred, ref, 0, 10)
        aligned = pred @ r.T + tr
        np.testing.assert_allclose(aligned[0], ref[0])
        self.assertAlmostEqual(np.linalg.det(r), 1.)
        self.assertAlmostEqual(np.linalg.norm(aligned[-1] - ref[-1]), 180.)

if __name__ == "__main__":
    unittest.main()
