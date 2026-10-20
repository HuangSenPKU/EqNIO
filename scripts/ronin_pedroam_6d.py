#!/usr/bin/env python3
"""Frozen EqNIO RoNIN inference with magnetometer-free attitude on PedROAM.

Only total acceleration and calibrated gyro enter the estimator. Reference
positions are opened after prediction, solely for offline rigid XY evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
sys.path.insert(0, str(ROOT / "RONIN/source"))

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation, Slerp
from attitude_6d import madgwick_orientation


MODELS = {
    "o2": ("Ronin_o2", "checkpoint_38.pt"),
}
SENSORS = {
    "gyro": "Gyroscope.csv",
    "accel": "TotalAcceleration.csv",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_sensor(path):
    frame = pd.read_csv(path, usecols=["time", "x", "y", "z"])
    ns = frame.time.to_numpy(np.int64)
    values = frame[["x", "y", "z"]].to_numpy(np.float64)
    if len(ns) < 2 or np.any(np.diff(ns) <= 0) or not np.isfinite(values).all():
        raise ValueError(f"Invalid timestamps or measurements: {path}")
    return ns, values


def interpolate(t_new, t_old, values):
    return np.ascontiguousarray(np.column_stack([
        np.interp(t_new, t_old, values[:, j]) for j in range(values.shape[1])
    ]))


def estimate_orientation(gyro, accel, gain=.033):
    """Madgwick 6D at 100 Hz on calibrated gyro; no extra bias subtraction."""
    return madgwick_orientation(gyro, accel, gain)


def prepare(folder, warmup_s=10.0, max_seconds=None, gain=.033):
    raw = {key: read_sensor(folder / name) for key, name in SENSORS.items()}
    origin = max(int(ts[0]) for ts, _ in raw.values())
    end = min(int(ts[-1]) for ts, _ in raw.values())
    full_end = end
    if max_seconds is not None:
        end = min(end, origin + round(max_seconds * 1e9))
    ns100 = np.arange(origin, end + 1, 10_000_000, dtype=np.int64)
    if len(ns100) < (warmup_s + 2) * 100:
        raise ValueError("Insufficient IMU duration after warmup")
    t100 = (ns100 - origin) * 1e-9
    data, quality = {}, {}
    for key, (ts, values) in raw.items():
        dt = np.diff(ts) * 1e-9
        relevant = (ts[:-1] < ns100[-1]) & (ts[1:] > origin)
        if np.any(dt[relevant] > 0.1):
            raise ValueError(f"{key}: gap >100 ms; split recording before inference")
        data[key] = interpolate(t100, (ts - origin) * 1e-9, values)
        quality[key] = {
            "rows": len(ts), "median_hz": float(1 / np.median(dt)),
            "max_gap_s": float(dt.max()),
        }
    state = estimate_orientation(data["gyro"], data["accel"], gain)
    q = state["quat6D"]
    rotations100 = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    ns200 = np.arange(origin, ns100[-1] + 1, 5_000_000, dtype=np.int64)
    t200 = (ns200 - origin) * 1e-9
    rotations200 = Slerp(t100, rotations100)(t200)
    imu = np.column_stack([
        rotations200.apply(interpolate(t200, t100, data["gyro"])),
        rotations200.apply(interpolate(t200, t100, data["accel"])),
    ]).astype(np.float32)
    # Exactly one second of displacement labels spans 201 timestamps, while
    # the official network consumes 200 samples. Retain the endpoint timestamp.
    first = int(np.ceil(warmup_s * 200 / 10)) * 10
    starts = np.arange(first, len(ns200) - 200, 10, dtype=np.int64)
    if len(starts) < 2:
        raise ValueError("Insufficient prediction windows")
    quality.update({
        "input_files": list(SENSORS.values()),
        "imu_common_duration_s": float((full_end - origin) * 1e-9),
        "processed_duration_s": float(t100[-1]),
        "warmup_s": warmup_s, "attitude": "madgwick",
        "madgwick_gain": gain,
        "gyro_bias_estimator": "none",
        "initialization": "Madgwick first accelerometer tilt, arbitrary yaw",
        "rest_detection_available": False,
        "quat_norm_max_error": float(np.max(abs(np.linalg.norm(q, axis=1) - 1))),
        "interpolation": "100 Hz linear IMU; causal attitude; 200 Hz linear IMU + quaternion SLERP",
    })
    diagnostics = {
        "timestamp_ns": ns100, "quat_wxyz": q,
    }
    return ns200, imu, starts, quality, diagnostics


def fit_yaw_translation(pred, reference):
    """Least-squares proper 2D rotation and translation; no scale/reflection."""
    x, y = pred - pred.mean(axis=0), reference - reference.mean(axis=0)
    dot = np.sum(x * y)
    cross = np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
    if np.linalg.norm(x) < 1e-8 or np.hypot(dot, cross) < 1e-8:
        raise ValueError("Trajectory has insufficient motion for yaw alignment")
    angle = float(np.arctan2(cross, dot))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    translation = reference.mean(axis=0) - pred.mean(axis=0) @ rotation.T
    return rotation, translation, angle


def geodetic_to_enu(lat_deg, lon_deg, height):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a, e2 = 6378137.0, 6.69437999014e-3
    radius = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    ecef = np.column_stack([
        (radius + height) * np.cos(lat) * np.cos(lon),
        (radius + height) * np.cos(lat) * np.sin(lon),
        (radius * (1 - e2) + height) * np.sin(lat),
    ])
    sl, cl, sp, cp = np.sin(lon[0]), np.cos(lon[0]), np.sin(lat[0]), np.cos(lat[0])
    rotation = np.array([[-sl, cl, 0], [-sp * cl, -sp * sl, cp], [cp * cl, cp * sl, sp]])
    return (ecef - ecef[0]) @ rotation.T


def evaluate(reference_path, ns, positions, velocity, out, attitude_label="Madgwick 6D"):
    """Ground truth is consumed only here, after local inference is saved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = pd.read_csv(reference_path)
    columns = ["observation_utc_seconds", "gt_latitude", "gt_longitude", "gt_height"]
    valid = gt.gt_valid.astype(str).str.lower().eq("true")
    valid &= np.isfinite(gt[columns].to_numpy()).all(axis=1)
    gt = gt.loc[valid].sort_values(columns[0]).drop_duplicates(columns[0])
    total_gt = len(gt)
    gt_ns = np.rint(gt[columns[0]].to_numpy() * 1e9).astype(np.int64)
    overlap = (gt_ns >= ns[0]) & (gt_ns <= ns[-1])
    gt, gt_ns = gt.loc[overlap], gt_ns[overlap]
    if len(gt) < 3:
        raise ValueError("Insufficient valid ground-truth overlap")
    t, tg = (ns - ns[0]) * 1e-9, (gt_ns - ns[0]) * 1e-9
    pred = interpolate(tg, t, positions)
    ref = geodetic_to_enu(
        gt.gt_latitude.to_numpy(), gt.gt_longitude.to_numpy(), gt.gt_height.to_numpy()
    )[:, :2]
    rotation, translation, angle = fit_yaw_translation(pred, ref)
    aligned = pred @ rotation.T + translation
    error = np.linalg.norm(aligned - ref, axis=1)
    unaligned = (pred - pred[0]) - (ref - ref[0])
    # Endpoint differences only; require continuous GT coverage for each pair.
    right = np.clip(np.searchsorted(tg, tg + 60), 1, len(tg) - 1)
    left = right - 1
    j = np.where(abs(tg[left] - tg - 60) < abs(tg[right] - tg - 60), left, right)
    i = np.arange(len(tg))
    gaps = np.r_[0, np.cumsum(np.diff(tg) > 2.0)]
    pairs = (abs(tg[j] - tg - 60) <= .1) & (gaps[j] == gaps[i])
    relative = np.linalg.norm(
        (aligned[j[pairs]] - aligned[i[pairs]]) - (ref[j[pairs]] - ref[i[pairs]]), axis=1
    )
    metrics = {
        "alignment": "full-overlap least-squares constant yaw + translation; no scale or reflection",
        "alignment_yaw_deg": float(np.rad2deg(angle)),
        "alignment_translation_m": translation.tolist(),
        "reference_samples": len(gt), "valid_reference_samples_total": total_gt,
        "reference_coverage": len(gt) / total_gt,
        "reference_duration_s": float(tg[-1] - tg[0]),
        "horizontal_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "horizontal_median_m": float(np.median(error)),
        "horizontal_p95_m": float(np.quantile(error, .95)),
        "final_error_m": float(error[-1]),
        "relative_60s_pairs": int(pairs.sum()),
        "relative_60s_rmse_m": float(np.sqrt(np.mean(relative ** 2))) if len(relative) else None,
        "translation_only_rmse_m": float(np.sqrt(np.mean(np.sum(unaligned ** 2, axis=1)))),
        "reference_path_m": float(np.linalg.norm(np.diff(ref, axis=0), axis=1)[np.diff(tg) <= 2].sum()),
        "reference_max_gap_s": float(np.diff(tg).max()),
    }
    pd.DataFrame({
        "timestamp_ns": gt_ns, "pred_local_x_m": pred[:, 0], "pred_local_y_m": pred[:, 1],
        "aligned_e_m": aligned[:, 0], "aligned_n_m": aligned[:, 1],
        "gt_e_m": ref[:, 0], "gt_n_m": ref[:, 1], "horizontal_error_m": error,
    }).to_csv(out / "evaluation.csv", index=False)
    all_aligned = positions @ rotation.T + translation
    world_velocity = velocity @ rotation.T
    pd.DataFrame({
        "timestamp_ns": ns, "aligned_e_m": all_aligned[:, 0], "aligned_n_m": all_aligned[:, 1],
        "aligned_ve_mps": world_velocity[:, 0], "aligned_vn_mps": world_velocity[:, 1],
    }).to_csv(out / "trajectory_aligned_offline.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(ref[:, 0], ref[:, 1], label="Reference", lw=1)
    axes[0].plot(aligned[:, 0], aligned[:, 1], label=f"EqNIO + {attitude_label}", lw=1)
    axes[0].set(xlabel="East (m)", ylabel="North (m)", aspect="equal",
                title="Offline constant yaw + translation; no scale")
    axes[0].legend()
    axes[1].plot((tg - tg[0]) / 60, error, lw=1)
    axes[1].set(xlabel="Time (min)", ylabel="Horizontal error (m)",
                title=f"RMSE {metrics['horizontal_rmse_m']:.2f} m")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out / "trajectory.png", dpi=160)
    plt.close(fig)
    return metrics


def load_models(checkpoint_root, variants, device):
    import torch
    import ronin_resnet as ronin
    loaded = {}
    for variant in variants:
        folder, filename = MODELS[variant]
        config_path = checkpoint_root / folder / "config.json"
        config = json.loads(config_path.read_text())
        if config["window_size"] != 200:
            raise ValueError("Only released 200-sample checkpoints are supported")
        checkpoint_path = checkpoint_root / folder / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = ronin.get_model(config["arch"]).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        preprocess = ronin.preprocess_eq_o2_frame
        loaded[variant] = (model, preprocess, {
            "architecture": config["arch"], "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
            "training_config": config, "config_sha256": sha256(config_path),
        })
    return loaded


def infer(model, preprocess, imu, starts, device, batch_size):
    import torch
    outputs = []
    check_error = None
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            selected = starts[offset:offset + batch_size]
            features = torch.from_numpy(np.stack([imu[s:s + 200].T for s in selected])).to(device)
            _, velocity = model(*preprocess(features))
            if check_error is None:
                # Check a fixed change of horizontal reference frame on real inputs.
                c, s = np.cos(.7), np.sin(.7)
                rotation = features.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                rotated = features[:4].clone()
                rotated[:, :3] = rotation @ rotated[:, :3]
                rotated[:, 3:] = rotation @ rotated[:, 3:]
                _, second = model(*preprocess(rotated))
                expected = velocity[:4] @ rotation[:2, :2].T
                torch.testing.assert_close(second, expected, rtol=2e-3, atol=2e-4)
                check_error = float((second - expected).abs().max().cpu())
            outputs.append(velocity.cpu().numpy())
            if offset % (batch_size * 100) == 0:
                print(f"  inference {min(offset + batch_size, len(starts))}/{len(starts)}", flush=True)
    predictions = np.concatenate(outputs)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite velocity predictions")
    return predictions, check_error


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/maxson/data/PedROAM"))
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT.parent / "outputs/eqnio-pretrained")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/ronin-current/inference")
    parser.add_argument("--madgwick-gain", type=float, default=.033)
    parser.add_argument("--session")
    parser.add_argument("--person")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.warmup_seconds < 0 or args.batch_size < 1:
        parser.error("warmup must be nonnegative and batch size positive")
    if not np.isfinite(args.madgwick_gain) or args.madgwick_gain <= 0:
        parser.error("Madgwick gain must be finite and positive")
    if args.out_root.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("Outputs must be outside the dataset")
    # Require a fresh run directory: never mix old summaries and new predictions.
    args.out_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models = load_models(args.checkpoint_root, ["o2"], device)
    method = f"madgwick6d-b{args.madgwick_gain:g}-nonebias"
    attitude_label = f"Madgwick 6D beta={args.madgwick_gain:g}"
    manifest = {
        "argv": sys.argv, "script_sha256": sha256(__file__), "device": str(device),
        "attitude_implementation_sha256": sha256(Path(__file__).with_name("attitude_6d.py")),
        "versions": {p: importlib.metadata.version(p) for p in ["ahrs", "numba", "numpy", "scipy", "torch", "pandas"]},
        "method": method, "orientation": attitude_label + ", gyro+accel only, 100 Hz",
        "input_files": list(SENSORS.values()),
        "madgwick_gain": args.madgwick_gain,
        "bias_estimator": "none",
        "network_hz": 200, "prediction_hz": 20, "window_samples": 200,
        "gyro": "Gyroscope.csv (phone calibrated), no additional bias subtraction",
        "acceleration": "total acceleration including gravity, m/s^2",
        "timestamp": "one-second window midpoint; trapezoidal velocity integration",
        "evaluation": "GT used only after inference; one fitted yaw and translation per recording/model",
        "limitations": [
            "Offline yaw fitting uses the full reference overlap; this is not known true-north initialization.",
            "100 Hz input interpolated to 200 Hz adds no sensor bandwidth.",
            "Yaw drift remains unobservable from gravity; no magnetic or GNSS corrections.",
            "Ten-second warmup is a fixed exclusion, not proof of filter convergence.",
            "Attitude filtering is causal, but input interpolation and window inference require buffering.",
        ],
    }
    write_json(args.out_root / "run.json", manifest)
    rows, failures = [], []
    folders = sorted((args.data_root / "recordings").glob("*/phones/*/sensorlogger"))
    selected = [f for f in folders if (not args.session or f.parts[-4] == args.session)
                and (not args.person or f.parts[-2] == args.person)]
    if not selected:
        raise ValueError("No matching recordings")
    for folder in selected:
        session, person = folder.parts[-4], folder.parts[-2]
        print(f"{session}/{person}: preparing six-axis IMU", flush=True)
        try:
            hashes = {name: sha256(folder / name) for name in SENSORS.values()}
            ns, imu, starts, quality, diagnostics = prepare(
                folder, args.warmup_seconds, args.max_seconds,
                args.madgwick_gain,
            )
            common = args.out_root / session / person
            common.mkdir(parents=True)
            write_json(common / "input_quality.json", quality)
            np.savez_compressed(common / "orientation_6d.npz", **diagnostics)
            for variant, (model, preprocess, checkpoint_info) in models.items():
                start_time = time.monotonic()
                out = common / f"eqnio-ronin-{variant}-{method}"
                out.mkdir()
                print(f"{session}/{person}/{variant}: {len(starts)} windows", flush=True)
                velocity, equivariance_error = infer(model, preprocess, imu, starts, device, args.batch_size)
                pred_ns = ns[starts] + 500_000_000
                t = (pred_ns - pred_ns[0]) * 1e-9
                position = cumulative_trapezoid(velocity, t, axis=0, initial=0)
                pd.DataFrame({
                    "timestamp_ns": pred_ns, "local_x_m": position[:, 0], "local_y_m": position[:, 1],
                    "local_vx_mps": velocity[:, 0], "local_vy_mps": velocity[:, 1],
                }).to_csv(out / "trajectory_local.csv", index=False)
                reference = args.data_root / "recordings" / session / "reference" / person / "trajectory-gt.csv"
                reference_hash = sha256(reference)
                metrics = evaluate(reference, pred_ns, position, velocity, out, attitude_label)
                unchanged = all(sha256(folder / name) == digest for name, digest in hashes.items())
                unchanged &= sha256(reference) == reference_hash
                if not unchanged:
                    raise RuntimeError("Input file changed during the run")
                summary = {
                    "session": session, "person": person, "variant": variant, "method": method,
                    **checkpoint_info, "quality": quality, "metrics": metrics,
                    "predictions": len(pred_ns), "input_sha256": hashes,
                    "reference_sha256": reference_hash, "inputs_unchanged": unchanged,
                    "rotation_equivariance_max_error_mps": equivariance_error,
                    "elapsed_seconds": time.monotonic() - start_time,
                }
                write_json(out / "summary.json", summary)
                rows.append({"session": session, "person": person, "variant": variant,
                             **{k: v for k, v in metrics.items() if not isinstance(v, (list, str))}})
                pd.DataFrame(rows).to_csv(args.out_root / "summary.csv", index=False)
                print(f"  RMSE={metrics['horizontal_rmse_m']:.3f} m, coverage={metrics['reference_coverage']:.3f}", flush=True)
        except Exception as exc:
            failures.append({"session": session, "person": person, "error": repr(exc)})
            print(f"FAILED {session}/{person}: {exc!r}", flush=True)
        write_json(args.out_root / "failures.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)} recording(s) failed; inspect failures.json")
    print(f"Completed {len(rows)} evaluations: {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
