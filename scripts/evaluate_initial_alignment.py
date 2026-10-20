#!/usr/bin/env python3
"""Anchor saved trajectories using 10 metres of continuous GT travel.

GT contains positions, not device attitude. Heading here means direction of
travel accumulated over the initial distance, not a measured phone yaw. The transform
is frozen after initialization; later reference positions only score errors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reference_quality import (
    reference_continuity, continuous_distance_interval,
    DEFAULT_MAX_GAP_S, DEFAULT_MAX_SPEED_MPS, DEFAULT_SETTLING_S,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_transform(pred, ref, i, j):
    """Anchor at i and use displacement from i to j; no scale."""
    x, y = pred[j] - pred[i], ref[j] - ref[i]
    lengths = np.array([np.linalg.norm(x), np.linalg.norm(y)])
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("Initial direction is undefined: zero or nonfinite displacement")
    angle = np.arctan2(x[0] * y[1] - x[1] * y[0], np.dot(x, y))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    translation = ref[i] - pred[i] @ rotation.T
    return rotation, translation, float(angle)


def cumulative_distance_index(ref, distance_m=10.):
    """First sample reaching the requested GT path length from the initial point."""
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Initialization distance must be finite and positive")
    if len(ref) < 2:
        raise ValueError("At least two matched reference positions are required")
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    reached = np.flatnonzero(path >= distance_m)
    if not len(reached):
        raise ValueError(f"Reference path never reaches {distance_m:g} metres")
    return int(reached[0])


def cumulative_distance_transform(pred, ref, distance_m=10.):
    """Accumulate displacement vectors over the same initial GT-distance interval."""
    j = cumulative_distance_index(ref, distance_m)
    # The sum of successive displacement vectors equals endpoint minus start.
    return initial_transform(pred, ref, 0, j)


def rmse(error):
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


def relative_rmse(t, error, after_s, segments=None):
    right = np.clip(np.searchsorted(t, t + 60), 1, len(t) - 1)
    left = right - 1
    j = np.where(abs(t[left] - t - 60) < abs(t[right] - t - 60), left, right)
    i = np.arange(len(t))
    gaps = np.r_[0, np.cumsum(np.diff(t) > 2.)]
    keep = (t > after_s) & (abs(t[j] - t - 60) <= .1) & (gaps[j] == gaps[i])
    if segments is not None:
        keep &= segments[j] == segments[i]
    return rmse(error[j[keep]] - error[i[keep]]) if keep.any() else None


def evaluate_runs(runs, out, initialization_distance_m=10., protocol="continuous",
                  max_gap_s=DEFAULT_MAX_GAP_S,
                  max_speed_mps=DEFAULT_MAX_SPEED_MPS,
                  settling_s=DEFAULT_SETTLING_S, min_displacement_m=.5):
    if protocol not in {"continuous", "legacy"}:
        raise ValueError("Unknown alignment protocol")
    if not np.isfinite(min_displacement_m) or min_displacement_m <= 0:
        raise ValueError("Minimum displacement must be finite and positive")
    quality_options = dict(max_gap_s=max_gap_s, max_speed_mps=max_speed_mps,
                           settling_s=settling_s)
    out.mkdir(parents=True, exist_ok=False)
    rows, calibration, checks = [], {}, []
    for label, root in runs:
        manifest = json.loads((root / "run.json").read_text())
        method = manifest["method"]
        assert manifest["bias_estimator"] == {"ronin": "none", "tlio": "ekf"}[manifest.get("model_family", "ronin")]
        assert set(manifest["input_files"]) == {"Gyroscope.csv", "TotalAcceleration.csv"}
        table = pd.read_csv(root / "summary.csv")
        table = table[table.variant == "o2"].sort_values(["session", "person"])
        keys = list(zip(table.session, table.person))
        if calibration:
            assert keys == list(calibration), "Sequence sets differ"
        for row in table.itertuples():
            key = (row.session, row.person)
            source = root / row.session / row.person / manifest.get("trajectory_subdir", f"eqnio-ronin-o2-{method}")
            provenance = json.loads((source / "summary.json").read_text())
            ev = pd.read_csv(source / "evaluation.csv")
            ns = ev.timestamp_ns.to_numpy(np.int64)
            t = (ns - ns[0]) * 1e-9
            pred = ev[["pred_local_x_m", "pred_local_y_m"]].to_numpy()
            ref = ev[["gt_e_m", "gt_n_m"]].to_numpy()
            quality = reference_continuity(t, ref, **quality_options)
            def select_interval(reference_t, reference_xy):
                if protocol == "legacy":
                    return 0, cumulative_distance_index(reference_xy, initialization_distance_m)
                return continuous_distance_interval(reference_t, reference_xy,
                                                    initialization_distance_m, **quality_options)
            identity = {k: provenance[k] for k in [
                "input_sha256", "reference_sha256"
            ]}
            if key not in calibration:
                i, j = select_interval(t, ref)
                calibration[key] = (i, j, ns, ref, identity)
            else:
                i, j, expected_ns, expected_ref, expected_identity = calibration[key]
                np.testing.assert_array_equal(ns, expected_ns)
                np.testing.assert_allclose(ref, expected_ref, atol=1e-9, rtol=0)
                assert identity == expected_identity
            if protocol == "continuous" and min(np.linalg.norm(pred[j] - pred[i]),
                                                np.linalg.norm(ref[j] - ref[i])) < min_displacement_m:
                raise ValueError(f"{label}/{row.session}/{row.person}: Initial direction is undefined: "
                                 f"net displacement < {min_displacement_m:g} m in the GT-selected interval; "
                                 "no model-dependent interval reselection")
            rotation, translation, angle = initial_transform(pred, ref, i, j)
            aligned = pred @ rotation.T + translation
            np.testing.assert_allclose(aligned[i], ref[i], atol=1e-9)
            np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
            # Verify future reference changes cannot alter the chosen transform.
            assert select_interval(t[:j + 1], ref[:j + 1]) == (i, j)
            prefix_r, prefix_tr, _ = initial_transform(pred[:j + 1], ref[:j + 1], i, j)
            np.testing.assert_array_equal(rotation, prefix_r)
            np.testing.assert_array_equal(translation, prefix_tr)
            error = aligned - ref
            old_error = ev[["aligned_e_m", "aligned_n_m"]].to_numpy() - ref
            full_rmse = rmse(error)
            assert full_rmse + 1e-6 >= rmse(old_error), "Global least squares must be optimal"
            post = t > t[j]
            if not post.any():
                raise ValueError(f"{key}: No samples after initialization; evaluate a longer recording")
            values = {
                "estimator": label, "session": row.session, "person": row.person,
                "initialization_start_s": float(t[i]), "initialization_end_s": float(t[j]),
                "initialization_start_timestamp_ns": int(ns[i]),
                "initialization_end_timestamp_ns": int(ns[j]),
                "initialization_target_distance_m": initialization_distance_m,
                "alignment_protocol": protocol,
                "initialization_gt_path_m": float(np.linalg.norm(np.diff(ref[i:j + 1], axis=0), axis=1).sum()),
                "initialization_gt_path_before_last_m": float(np.linalg.norm(np.diff(ref[i:j], axis=0), axis=1).sum()),
                "initialization_max_gt_gap_s": float(np.max(np.diff(t[i:j + 1]))),
                "gt_discontinuities": int(quality["gt_discontinuity"].sum()),
                "gt_speed_flags": int(quality["gt_speed_flag"].sum()),
                "gt_gap_flags": int(quality["gt_gap_flag"].sum()),
                "pre_initialization_discontinuities": int(quality["gt_discontinuity"][:j + 1].sum()),
                "heading_rotation_deg": float(np.rad2deg(angle)),
                "initial_heading_pred_displacement_m": float(np.linalg.norm(pred[j] - pred[i])),
                "initial_heading_gt_displacement_m": float(np.linalg.norm(ref[j] - ref[i])),
                "reference_samples": len(ev), "post_initialization_samples": int(post.sum()),
                "duration_s": float(t[-1]),
                "initial_full_rmse_m": full_rmse,
                "initial_post_rmse_m": rmse(error[post]),
                "initial_first120_rmse_m": rmse(error[t <= 120.]),
                "initial_final_error_m": float(np.linalg.norm(error[-1])),
                "initial_post_relative60_rmse_m": relative_rmse(
                    t, error, t[j], quality["gt_segment"] if protocol == "continuous" else None),
                "offline_full_rmse_m": rmse(old_error),
                "offline_post_rmse_m": rmse(old_error[post]),
            }
            rows.append(values)
            target = out / label / row.session / row.person
            target.mkdir(parents=True)
            (target / "summary.json").write_text(json.dumps({
                **values, "source": str(source.resolve()),
                "rotation": rotation.tolist(), "translation": translation.tolist(),
                "position_anchor_timestamp_ns": int(ns[i]),
                "quality_options": quality_options,
                "minimum_net_displacement_m": min_displacement_m if protocol == "continuous" else None,
                "source_evaluation_sha256": digest(source / "evaluation.csv"),
                "source_trajectory_sha256": digest(source / "trajectory_local.csv"),
            }, indent=2) + "\n")
            pd.DataFrame({
                "timestamp_ns": ns, "elapsed_s": t,
                "initial_aligned_e_m": aligned[:, 0], "initial_aligned_n_m": aligned[:, 1],
                "gt_e_m": ref[:, 0], "gt_n_m": ref[:, 1],
                "horizontal_error_m": np.linalg.norm(error, axis=1),
                "post_initialization": post,
                **quality,
            }).to_csv(target / "evaluation.csv", index=False)
            local = pd.read_csv(source / "trajectory_local.csv")
            xy = local[["local_x_m", "local_y_m"]].to_numpy() @ rotation.T + translation
            velocity = local[["local_vx_mps", "local_vy_mps"]].to_numpy() @ rotation.T
            pd.DataFrame({
                "timestamp_ns": local.timestamp_ns, "east_m": xy[:, 0], "north_m": xy[:, 1],
                "velocity_e_mps": velocity[:, 0], "velocity_n_mps": velocity[:, 1],
                "initialization_complete": local.timestamp_ns > ns[j],
            }).to_csv(target / "trajectory_initial_aligned.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            axes[0].plot(ref[:, 0], ref[:, 1], label="GT", lw=1)
            axes[0].plot(aligned[:, 0], aligned[:, 1], label="Initial alignment", lw=1)
            axes[0].plot(ref[i:j + 1, 0], ref[i:j + 1, 1], lw=3, label="Heading initialization")
            axes[0].set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
            axes[0].legend(fontsize=8)
            axes[1].plot(t / 60, np.linalg.norm(error, axis=1), label="Initial alignment")
            axes[1].plot(t / 60, np.linalg.norm(old_error, axis=1), label="Offline full-trajectory fit", alpha=.7)
            axes[1].axvline(t[j] / 60, color="gray", linestyle="--")
            axes[1].set(xlabel="Time (minutes)", ylabel="Horizontal error (m)")
            axes[1].legend(fontsize=8)
            for ax in axes:
                ax.grid(alpha=.25)
            fig.suptitle(f"{label} | {row.session}/{row.person} | Post-init RMSE {values['initial_post_rmse_m']:.1f} m")
            fig.savefig(target / "trajectory.png", dpi=140)
            plt.close(fig)
            checks.append({"estimator": label, "session": row.session, "person": row.person,
                           "initial_position_exact": True, "prefix_only_transform": True,
                           "proper_rotation_no_scale": True})
    result = pd.DataFrame(rows)
    result.to_csv(out / "per_sequence.csv", index=False)
    aggregate = result.groupby("estimator", sort=False).agg(
        sequences=("session", "size"),
        mean_initial_full_rmse_m=("initial_full_rmse_m", "mean"),
        mean_initial_post_rmse_m=("initial_post_rmse_m", "mean"),
        median_initial_post_rmse_m=("initial_post_rmse_m", "median"),
        mean_initial_post_relative60_rmse_m=("initial_post_relative60_rmse_m", "mean"),
        mean_offline_full_rmse_m=("offline_full_rmse_m", "mean"),
    ).reset_index()
    aggregate.to_csv(out / "aggregate.csv", index=False)
    (out / "run.json").write_text(json.dumps({
        "script_sha256": digest(Path(__file__)),
        "reference_quality_sha256": digest(Path(__file__).with_name("reference_quality.py")),
        "runs": {label: str(root.resolve()) for label, root in runs},
        "variant": "o2", "protocol": protocol,
        "quality_options": quality_options,
        "minimum_net_displacement_m": min_displacement_m if protocol == "continuous" else None,
        "position_anchor": "reference position at selected initialization start",
        "heading": "angle(sum GT displacement vectors) - angle(sum predicted displacement vectors); not measured device yaw",
        "initialization_distance_m": initialization_distance_m,
        "selection": ("first continuous interval reaching requested GT path length; reset at flagged gaps/speeds; "
                      "wait fixed settling time after each break before choosing a new anchor"
                      if protocol == "continuous" else
                      "legacy: first overlapping valid GT point to first accumulated-distance crossing, including gaps"),
        "distance_details": "accumulated path length, not net displacement; no smoothing or straightness selection; "
                            "selection uses GT and timestamps only; no model-dependent retries",
        "comparison": "same initialization interval for every estimator per recording",
        "causality": "transform available only at initialization end; earlier points retroactively transformed",
        "post_initialization": "strictly after initialization end; primary evaluation excludes calibration data",
        "reference_policy": "coordinates and post-init position samples retained unchanged; continuity flags are "
                            "suspicions, not certified invalid positions; no reference-frame repair",
        "relative60": ("exclude every pair crossing a flagged discontinuity, plus legacy >2 s gap rule"
                       if protocol == "continuous" else "legacy: exclude >2 s reference gaps"),
        "correction": "one fixed proper yaw rotation and selected-start translation; no scale or later updates",
        "network": "reuses unchanged saved local inference; no magnetic input and no GT fed into model",
    }, indent=2) + "\n")
    (out / "verification.json").write_text(json.dumps({
        "evaluations": len(checks), "failures": 0, "checks": checks,
    }, indent=2) + "\n")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "outputs/ronin-current"
    parser.add_argument("--result-root", type=Path, default=root / "inference")
    parser.add_argument("--out-root", type=Path, default=root / "alignment-10m-continuous")
    parser.add_argument("--initialization-distance-m", type=float, default=10.)
    parser.add_argument("--label", default="Madgwick")
    parser.add_argument("--protocol", choices=["continuous", "legacy"], default="continuous")
    parser.add_argument("--max-gt-gap-s", type=float, default=DEFAULT_MAX_GAP_S)
    parser.add_argument("--max-gt-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS)
    parser.add_argument("--settling-s", type=float, default=DEFAULT_SETTLING_S)
    parser.add_argument("--min-displacement-m", type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.initialization_distance_m) or args.initialization_distance_m <= 0:
        parser.error("Initialization distance must be finite and positive")
    evaluate_runs([(args.label, args.result_root)], args.out_root, args.initialization_distance_m,
                  args.protocol, args.max_gt_gap_s, args.max_gt_speed_mps,
                  args.settling_s, args.min_displacement_m)
