#!/usr/bin/env python3
"""Independently verify saved inference results and create comparison artifacts."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(root):
    manifest = json.loads((root / "run.json").read_text())
    method = manifest["method"]
    expected_inputs = {"Gyroscope.csv", "TotalAcceleration.csv"}
    assert set(manifest["input_files"]) == expected_inputs
    summary = pd.read_csv(root / "summary.csv")
    assert json.loads((root / "failures.json").read_text()) == []
    assert not summary.duplicated(["session", "person", "variant"]).any()
    assert np.isfinite(summary.select_dtypes("number").drop(columns=["relative_60s_rmse_m"])).all().all()
    checks = []
    for row in summary.itertuples():
        folder = root / row.session / row.person / f"eqnio-ronin-{row.variant}-{method}"
        provenance = json.loads((folder / "summary.json").read_text())
        assert provenance["inputs_unchanged"]
        assert set(provenance["input_sha256"]) == expected_inputs
        ev = pd.read_csv(folder / "evaluation.csv")
        pred = ev[["pred_local_x_m", "pred_local_y_m"]].to_numpy()
        ref = ev[["gt_e_m", "gt_n_m"]].to_numpy()
        x, y = pred - pred.mean(0), ref - ref.mean(0)
        # Independent SVD solution, rather than inference's atan2 formulation.
        u, _, vh = np.linalg.svd(x.T @ y)
        row_rotation = u @ np.diag([1., np.linalg.det(u @ vh)]) @ vh
        expected = x @ row_rotation + ref.mean(0)
        aligned = ev[["aligned_e_m", "aligned_n_m"]].to_numpy()
        np.testing.assert_allclose(expected, aligned, atol=1e-8)
        errors = np.linalg.norm(aligned - ref, axis=1)
        np.testing.assert_allclose(errors, ev.horizontal_error_m, atol=1e-8)
        rmse = np.sqrt(np.mean(errors ** 2))
        np.testing.assert_allclose(rmse, row.horizontal_rmse_m)
        local = pd.read_csv(folder / "trajectory_local.csv")
        dt = np.diff(local.timestamp_ns.to_numpy(np.int64)) * 1e-9
        velocity = local[["local_vx_mps", "local_vy_mps"]].to_numpy()
        integrated = np.vstack([np.zeros(2), np.cumsum(
            .5 * (velocity[:-1] + velocity[1:]) * dt[:, None], axis=0
        )])
        # CSV roundtrips and float32 model velocities can differ at sub-micron
        # scale when independently reintegrated with float64 arithmetic.
        np.testing.assert_allclose(integrated, local[["local_x_m", "local_y_m"]], atol=1e-5)
        assert (folder / "trajectory.png").exists()
        checks.append({"session": row.session, "person": row.person,
                       "variant": row.variant, "rmse_verified_m": rmse})
    aggregate = []
    for variant, group in summary.groupby("variant"):
        relative = group.dropna(subset=["relative_60s_rmse_m"])
        aggregate.append({
            "variant": variant, "sequences": len(group),
            "mean_sequence_rmse_m": group.horizontal_rmse_m.mean(),
            "median_sequence_rmse_m": group.horizontal_rmse_m.median(),
            "pooled_position_rmse_m": np.sqrt(np.average(
                group.horizontal_rmse_m ** 2, weights=group.reference_samples)),
            "mean_sequence_relative_60s_rmse_m": relative.relative_60s_rmse_m.mean(),
            "pooled_relative_60s_rmse_m": np.sqrt(np.average(
                relative.relative_60s_rmse_m ** 2, weights=relative.relative_60s_pairs)) if len(relative) else np.nan,
            "min_coverage": group.reference_coverage.min(),
            "max_coverage": group.reference_coverage.max(),
        })
    aggregate = pd.DataFrame(aggregate)
    aggregate.to_csv(root / "aggregate.csv", index=False)
    (root / "verification.json").write_text(json.dumps({
        "evaluations": len(checks), "failures": 0,
        "verified": ["RMSE recomputed from saved trajectories",
                     "alignment independently recomputed with SVD",
                     "velocity reintegrated independently (10 micrometer tolerance)",
                     "two IMU inputs only in manifests",
                     "recorded consumed-input hashes unchanged", "all trajectory plots present"],
        "results": checks,
    }, indent=2) + "\n")
    sequences = summary[["session", "person"]].drop_duplicates()
    labels = [session.replace("-2026", "") + "/" + person
              for session, person in sequences.itertuples(index=False, name=None)]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(labels) * .55 + 1)), constrained_layout=True)
    y = np.arange(len(labels))
    variants = sorted(summary.variant.unique())
    width = .72 / len(variants)
    for ax, metric, title in zip(
        axes, ["horizontal_rmse_m", "relative_60s_rmse_m"],
        ["Horizontal position RMSE", "60-second relative displacement RMSE"]
    ):
        for k, variant in enumerate(variants):
            values = sequences.merge(summary[summary.variant == variant], on=["session", "person"])[metric]
            bars = ax.barh(y + (k - (len(variants) - 1) / 2) * width, values,
                           height=width * .95, label=variant.upper(),
                           color="#137c8b")
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set(xlabel="Error (m)", title=title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(x=.15)
        ax.legend()
    fig.suptitle(f"PedROAM | {method} + frozen EqNIO RoNIN\n"
                 "One offline yaw + translation per recording; no scale fitting", fontsize=14)
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    print(aggregate.to_string(index=False))
    print(summary.pivot(index=["session", "person"], columns="variant", values="horizontal_rmse_m").round(2))
    hours = sum(json.loads(p.read_text())["processed_duration_s"]
                for p in root.glob("*/*/input_quality.json")) / 3600
    print(f"Verified {len(checks)} evaluations; {hours:.3f} recording hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    summarize(parser.parse_args().result_root)
