"""Magnetometer-free Madgwick update, compiled for long recordings.

Equations match AHRS 0.4.0 Madgwick.updateIMU; reference agreement is tested.
https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
Quaternion convention: wxyz, sensor to gravity-aligned local frame.
"""
import numpy as np
from numba import njit
from ahrs.common.orientation import acc2q


@njit
def _madgwick_batch(gyro, accel, initial, gain, dt):
    result = np.empty((len(gyro), 4))
    result[0] = initial
    for i in range(1, len(gyro)):
        w, x, y, z = result[i - 1]
        gx, gy, gz = gyro[i]
        # Preserve the reference implementation's zero-angular-rate behavior.
        if gx * gx + gy * gy + gz * gz == 0:
            result[i] = result[i - 1]
            continue
        derivative = np.array([
            -.5 * (x * gx + y * gy + z * gz),
            .5 * (w * gx + y * gz - z * gy),
            .5 * (w * gy - x * gz + z * gx),
            .5 * (w * gz + x * gy - y * gx),
        ])
        ax, ay, az = accel[i]
        norm = np.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0:
            ax, ay, az = ax / norm, ay / norm, az / norm
            f0 = 2 * (x * z - w * y) - ax
            f1 = 2 * (w * x + y * z) - ay
            f2 = 1 - 2 * (x * x + y * y) - az
            gradient = np.array([
                -2 * y * f0 + 2 * x * f1,
                2 * z * f0 + 2 * w * f1 - 4 * x * f2,
                -2 * w * f0 + 2 * z * f1 - 4 * y * f2,
                2 * x * f0 + 2 * y * f1,
            ])
            norm_gradient = np.sqrt(np.sum(gradient ** 2))
            if norm_gradient > 1e-15:
                derivative -= gain * gradient / norm_gradient
        updated = result[i - 1] + dt * derivative
        result[i] = updated / np.sqrt(np.sum(updated ** 2))
    return result


def madgwick_orientation(gyro, accel, gain=.033):
    """Forward Madgwick IMU on calibrated gyro, without extra bias estimation."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Madgwick gain must be finite and positive")
    gyro = np.ascontiguousarray(gyro, dtype=np.float64)
    accel = np.ascontiguousarray(accel, dtype=np.float64)
    if np.linalg.norm(accel[0]) < 1e-8:
        raise ValueError("Initial acceleration cannot determine tilt")
    quaternion = _madgwick_batch(gyro, accel, acc2q(accel[0]), gain, .01)
    return {"quat6D": quaternion}
