#!/usr/bin/env python3
"""Export saved TLIO EKF marginal variances and standard deviations to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def write_uncertainty_csv(path, timestamp_ns, covariance_diagonal):
    """Write (N, 15) EKF error-state variances with integer epoch-ns times.

    State order: rotation error, velocity, position, gyro bias, accel bias.
    Rotation error/velocity/position use the filter's local frame, not aligned
    ENU; biases use IMU axes. Standard deviations are sqrt(variance), not
    empirical errors or confidence bounds calibrated against ground truth.
    """
    times = np.asarray(timestamp_ns)
    variances = np.asarray(covariance_diagonal)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("timestamp_ns must be a one-dimensional integer array")
    if variances.shape != (len(times), 15):
        raise ValueError("Expected covariance diagonal of shape (N, 15)")
    if not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("Covariance variances must be finite and nonnegative")
    groups = (
        ("local_rotation_error", "rad2", "rad"),
        ("local_velocity", "m2ps2", "mps"),
        ("local_position", "m2", "m"),
        ("gyro_bias", "rad2ps2", "radps"),
        ("accel_bias", "m2ps4", "mps2"),
    )
    columns = {"timestamp_ns": times}
    for group_index, (name, variance_unit, std_unit) in enumerate(groups):
        for axis_index, axis in enumerate("xyz"):
            variance = variances[:, 3 * group_index + axis_index]
            columns[f"{name}_{axis}_variance_{variance_unit}"] = variance
            columns[f"{name}_{axis}_std_{std_unit}"] = np.sqrt(variance)
    pd.DataFrame(columns).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/tlio-current/inference")
    args = parser.parse_args()
    files = sorted(args.result_root.rglob("filter_diagnostics.npz"))
    if not files:
        parser.error(f"No filter_diagnostics.npz found under {args.result_root}")
    for source in files:
        target = source.with_name("uncertainty_local.csv")
        with np.load(source) as data:
            write_uncertainty_csv(target, data["timestamp_ns"], data["state_covariance_diagonal"])
        print(target)
    print(f"Exported {len(files)} uncertainty CSV files")


if __name__ == "__main__":
    main()
