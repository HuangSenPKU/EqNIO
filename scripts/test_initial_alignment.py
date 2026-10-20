"""Verify initial-only alignment, including future-reference independence."""
import unittest
import numpy as np
from evaluate_initial_alignment import (
    initial_transform,
    cumulative_distance_index, cumulative_distance_transform, relative_rmse,
)
from reference_quality import reference_continuity, continuous_distance_interval


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


class ReferenceContinuityTests(unittest.TestCase):
    def test_jump_and_settling_cannot_supply_initial_distance(self):
        t = np.arange(80.)
        ref = np.column_stack([np.where(t < 5, 0., 37.) + np.maximum(t - 30, 0.),
                               np.zeros(len(t))])
        self.assertEqual(cumulative_distance_index(ref), 5)
        self.assertEqual(continuous_distance_interval(t, ref), (15, 40))
        quality = reference_continuity(t, ref)
        self.assertTrue(quality["gt_speed_flag"][5])
        self.assertFalse(quality["gt_anchor_ready"][14])
        self.assertTrue(quality["gt_anchor_ready"][15])

    def test_gap_resets_even_when_average_speed_is_plausible(self):
        t = np.r_[np.arange(5.), np.arange(7., 50.)]
        ref = np.column_stack([t, np.zeros(len(t))])
        i, j = continuous_distance_interval(t, ref)
        self.assertEqual((t[i], t[j]), (17., 27.))
        self.assertFalse(reference_continuity(t, ref)["gt_speed_flag"].any())

    def test_second_break_restarts_settling(self):
        t = np.arange(60.)
        x = t + 37 * (t >= 4) + 20 * (t >= 9)
        ref = np.column_stack([x, t * 0])
        self.assertEqual(continuous_distance_interval(t, ref), (19, 29))

    def test_clean_data_retains_cumulative_path_protocol(self):
        t = np.arange(20.)
        ref = np.column_stack([np.minimum(t, 6), np.maximum(t - 6, 0)])
        self.assertEqual(continuous_distance_interval(t, ref), (0, 10))
        self.assertLess(np.linalg.norm(ref[10] - ref[0]), 10)

    def test_reanchoring_and_future_independence(self):
        t = np.arange(60.)
        ref = np.column_stack([t + 37 * (t >= 5), np.zeros(len(t))])
        pred = np.column_stack([np.zeros(len(t)), t])
        i, j = continuous_distance_interval(t, ref)
        r, tr, _ = initial_transform(pred, ref, i, j)
        np.testing.assert_allclose(pred[i] @ r.T + tr, ref[i], atol=1e-10)
        # The corrupt earlier reference is not retained as a translation anchor.
        self.assertGreater(np.linalg.norm(pred[0] @ r.T + tr - ref[0]), 30)
        ref[j + 1:] += 1e6
        self.assertEqual(continuous_distance_interval(t, ref), (i, j))
        self.assertEqual(continuous_distance_interval(t[:j + 1], ref[:j + 1]), (i, j))
        r2, tr2, _ = initial_transform(pred, ref, i, j)
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_array_equal(tr, tr2)

    def test_no_eligible_segment_fails_explicitly(self):
        t = np.arange(30.)
        ref = np.column_stack([37 * (t >= 5), np.zeros(len(t))])
        with self.assertRaisesRegex(ValueError, "No continuous"):
            continuous_distance_interval(t, ref)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            reference_continuity(t[::-1], ref)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            reference_continuity(t, ref * np.nan)

    def test_relative_metric_does_not_cross_speed_jump(self):
        t = np.arange(130.)
        error = np.column_stack([100 * (t >= 65), np.zeros(len(t))])
        segments = (t >= 65).astype(int)
        self.assertGreater(relative_rmse(t, error, 0), 0)
        self.assertEqual(relative_rmse(t, error, 0, segments), 0)

if __name__ == "__main__":
    unittest.main()
