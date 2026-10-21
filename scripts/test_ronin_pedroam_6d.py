"""Numerical checks for six-axis orientation and rigid XY evaluation."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ronin_pedroam_6d import (
    estimate_orientation, fit_yaw_translation, prepare, export_initial_alignment,
)


class SixAxisTests(unittest.TestCase):
    def test_initial_export_without_evaluation_preserves_timestamps_and_rotates_velocity(self):
        ns = 1_770_000_000_000_000_000 + np.arange(601, dtype=np.int64) * 50_000_000
        t = (ns - ns[0]) * 1e-9
        position = np.column_stack([t, np.zeros(len(t))])
        velocity = np.tile([1., 0.], (len(t), 1))
        gt_ns = ns[::20]
        # A jump at 5 s forces the shared protocol to reanchor at 15 s.
        ref = np.column_stack([np.full(len(gt_ns), 12.),
                               np.arange(len(gt_ns)) + 37 * (np.arange(len(gt_ns)) >= 5)])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            reference = out / "reference.csv"
            reference.write_text("synthetic reference")
            pd.DataFrame({"timestamp_ns": ns, "local_x_m": position[:, 0],
                          "local_y_m": position[:, 1]}).to_csv(out / "trajectory_local.csv", index=False)
            with patch("ronin_pedroam_6d.matched_reference", return_value=(gt_ns, ref, len(ref))):
                result = export_initial_alignment(reference, ns, position, velocity, out)
            csv = pd.read_csv(out / "trajectory_initial_aligned.csv")
            np.testing.assert_array_equal(csv.timestamp_ns, ns)
            np.testing.assert_allclose(csv.east_m, 12., atol=1e-12)
            np.testing.assert_allclose(csv.north_m, t + 37., atol=1e-12)
            np.testing.assert_allclose(csv.velocity_e_mps, 0., atol=1e-12)
            np.testing.assert_allclose(csv.velocity_n_mps, 1., atol=1e-12)
            self.assertEqual(result["initialization_start_timestamp_ns"], int(gt_ns[15]))
            self.assertEqual(result["initialization_end_timestamp_ns"], int(gt_ns[25]))
            np.testing.assert_array_equal(csv.initialization_complete, ns > gt_ns[25])
            ref[26:] += 1e6
            with patch("ronin_pedroam_6d.matched_reference", return_value=(gt_ns, ref, len(ref))):
                future_changed = export_initial_alignment(reference, ns, position, velocity, out)
            self.assertEqual(result, future_changed)
            pd.testing.assert_frame_equal(csv, pd.read_csv(out / "trajectory_initial_aligned.csv"))
            self.assertFalse((out / "evaluation.csv").exists())

    def test_initial_export_rejects_insufficient_travel_without_aligned_output(self):
        ns = np.arange(20, dtype=np.int64) * 1_000_000_000
        position = np.zeros((len(ns), 2))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch("ronin_pedroam_6d.matched_reference", return_value=(ns, position, len(ns))):
                with self.assertRaisesRegex(ValueError, "No continuous"):
                    export_initial_alignment(out / "unused.csv", ns, position, position, out)
            self.assertFalse((out / "trajectory_initial_aligned.csv").exists())

    def test_tilted_stationary_sensor_aligns_up_without_magnetometer(self):
        acceleration = np.tile([3.0, 4.0, np.sqrt(9.81 ** 2 - 25)], (2000, 1))
        state = estimate_orientation(np.zeros_like(acceleration), acceleration)
        q = state["quat6D"][-1]
        rotated = Rotation.from_quat(q[[1, 2, 3, 0]]).apply(acceleration[-1])
        np.testing.assert_allclose(rotated, [0, 0, 9.81], atol=1e-3)
        self.assertNotIn("quat9D", state)

    def test_gyro_tracks_yaw_instead_of_resetting_each_frame(self):
        n = 1000
        gyro = np.tile([0.0, 0.0, .2], (n, 1))
        accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        q = estimate_orientation(gyro, accel)["quat6D"]
        yaw = np.unwrap(Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler("xyz")[:, 2])
        self.assertAlmostEqual(yaw[-1] - yaw[0], .2 * (n - 1) * .01, delta=.02)

    def test_compiled_madgwick_matches_reference(self):
        from ahrs.filters import Madgwick
        rng = np.random.default_rng(15)
        gyro = rng.normal(0, .4, (1000, 3))
        accel = rng.normal(0, .6, (1000, 3)) + [2, 3, 9]
        for gain in [.01, .033, .1]:
            expected = Madgwick(gyr=gyro, acc=accel, frequency=100, gain=gain).Q
            actual = estimate_orientation(gyro, accel, gain)["quat6D"]
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_alignment_recovers_rotation_translation_without_scale(self):
        pred = np.array([[0., 0.], [1., 0.], [2., 3.], [-1., 2.]])
        angle = .8
        expected = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        ref = pred @ expected.T + [4, -2]
        rotation, translation, recovered = fit_yaw_translation(pred, ref)
        np.testing.assert_allclose(rotation, expected, atol=1e-12)
        np.testing.assert_allclose(translation, [4, -2], atol=1e-12)
        self.assertAlmostEqual(recovered, angle)
        scaled_ref = 2 * pred @ expected.T + [4, -2]
        r, tr, _ = fit_yaw_translation(pred, scaled_ref)
        self.assertGreater(np.linalg.norm(pred @ r.T + tr - scaled_ref), 1)
        self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_preparation_needs_only_two_imu_files_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            n = 1501
            ns = 1_770_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 10_000_000
            gyro = pd.DataFrame({"time": ns, "z": np.zeros(n), "y": np.zeros(n), "x": np.zeros(n)})
            accel = pd.DataFrame({"time": ns, "z": np.full(n, 9.81), "y": np.zeros(n), "x": np.zeros(n)})
            gyro.to_csv(folder / "Gyroscope.csv", index=False)
            # This deliberately invalid file must never be opened by inference.
            (folder / "GyroscopeUncalibrated.csv").write_text("do not read this stream")
            accel.to_csv(folder / "TotalAcceleration.csv", index=False)
            time_ns, imu, starts, quality, diagnostics = prepare(folder)
            np.testing.assert_array_equal(np.diff(time_ns), 5_000_000)
            self.assertEqual(starts[0], 2000)
            np.testing.assert_allclose(imu[:, :3], 0, atol=1e-6)
            np.testing.assert_allclose(imu[:, 5], 9.81, atol=1e-5)
            self.assertEqual(set(quality["input_files"]), {"Gyroscope.csv", "TotalAcceleration.csv"})
            self.assertEqual(diagnostics["quat_wxyz"].shape, (n, 4))
            gyro.drop(index=np.arange(1100, 1120)).to_csv(
                folder / "Gyroscope.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "gap >100 ms"):
                prepare(folder)


if __name__ == "__main__":
    unittest.main()
