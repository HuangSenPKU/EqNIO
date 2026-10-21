#!/usr/bin/env python3
"""Frozen EqNIO-TLIO O2 + original stochastic-cloning EKF on calibrated PedROAM IMU.

Madgwick supplies only the initial attitude after 10 s. Subsequent attitude,
velocity, position and residual biases belong to TLIO's EKF. No GT is accessed
until local inference has finished. Evaluation uses the shared 10 m protocol.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Single-threaded small matrix operations avoid BLAS oversubscription in the EKF.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "outputs/.numba"))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
from export_tlio_uncertainty import write_uncertainty_csv
from ronin_pedroam_6d import (
    SENSORS, estimate_orientation, evaluate, interpolate, read_sensor, sha256, write_json,
)
sys.path.insert(0, str(ROOT / "TLIO-master/src"))
from tracker.imu_tracker import ImuTracker
from utils.logging import logging

ARCH = "eq_o2_frame_fullCov_2vec_2deep"
METHOD = "ekf-madgwick-init"
SUBDIR = f"eqnio-tlio-o2-{METHOD}"
# Match main_filter.py defaults explicitly (some ImuMSCKF class defaults differ).
FILTER_CONFIG = dict(
    g_norm=9.81, sigma_na=float(np.sqrt(1e-3)), sigma_ng=float(np.sqrt(1e-4)),
    ita_ba=1e-4, ita_bg=1e-6, init_attitude_sigma=float(np.pi / 180),
    init_yaw_sigma=float(.1 * np.pi / 180), init_vel_sigma=1., init_pos_sigma=.001,
    init_bg_sigma=.0001, init_ba_sigma=.02, meascov_scale=10.,
    use_const_cov=False, add_sim_meas_noise=False, mahalanobis_fail_scale=0.,
)


def prepare_body(folder, max_seconds=None):
    """Same 100 -> 200 Hz IMU interpolation and output grid as the RoNIN run."""
    raw = {k: read_sensor(folder / f) for k, f in SENSORS.items()}
    origin = max(int(ts[0]) for ts, _ in raw.values())
    end = min(int(ts[-1]) for ts, _ in raw.values())
    if max_seconds is not None:
        end = min(end, origin + round(max_seconds * 1e9))
    ns100 = np.arange(origin, end + 1, 10_000_000, dtype=np.int64)
    if len(ns100) < 1200:
        raise ValueError("Insufficient duration for 10 s warmup and network windows")
    t100 = (ns100 - origin) * 1e-9
    data, quality = {}, {}
    for k, (ts, values) in raw.items():
        dt = np.diff(ts) * 1e-9
        relevant = (ts[:-1] < ns100[-1]) & (ts[1:] > origin)
        if np.any(dt[relevant] > .1):
            raise ValueError(f"{k}: IMU gap >100 ms")
        data[k] = interpolate(t100, (ts-origin)*1e-9, values)
        quality[k] = dict(rows=len(ts), median_hz=float(1/np.median(dt)), max_gap_s=float(dt.max()))
    # Causal initial attitude: never compute or consume later Madgwick estimates.
    initial_q = estimate_orientation(data['gyro'][:1001], data['accel'][:1001], .033)['quat6D'][-1]
    initial_r = Rotation.from_quat(initial_q[[1, 2, 3, 0]]).as_matrix()
    ns = np.arange(origin, ns100[-1] + 1, 5_000_000, dtype=np.int64)
    t = (ns-origin)*1e-9
    gyro = interpolate(t, t100, data['gyro'])
    accel = interpolate(t, t100, data['accel'])
    starts = np.arange(2000, len(ns)-200, 10, dtype=np.int64)
    output_indices = starts + 100
    quality.update(processed_duration_s=float(t100[-1]), initial_quat_wxyz=initial_q.tolist(),
                   input_files=list(SENSORS.values()), warmup_s=10., madgwick_gain=.033,
                   initial_position_m=[0., 0., 0.], initial_velocity_mps=[0., 0., 0.],
                   initial_accel_bias=[0., 0., 0.], initial_gyro_bias=[0., 0., 0.])
    return ns, gyro, accel, initial_r, output_indices, quality


def reset_position_velocity(tracker):
    """Preserve upstream's non-VIO first-update reset (no reference data)."""
    state = tracker.filter.state
    zero = np.zeros((3, 1))
    tracker.filter.reset_state_and_covariance(
        state.si_Rs, [zero.copy() for _ in state.si_timestamps_us],
        state.s_R, zero.copy(), zero.copy(), state.s_ba, state.s_bg,
    )


def infer(ns, gyro, accel, initial_r, indices, checkpoint, parameters, cpu=False):
    tracker = ImuTracker(str(checkpoint), str(parameters), 20., SimpleNamespace(**FILTER_CONFIG),
                         imu_calib=None, force_cpu=cpu)
    tracker.callback_first_update = reset_position_velocity
    assert tracker.debug_callback_get_meas is None
    tracker.init_with_state_at_time(10_000_000, initial_r, np.zeros((3, 1)), np.zeros((3, 1)),
                                    gyro[2000, :, None], accel[2000, :, None])
    saved, debug, cursor = [], [], 0
    updates = 0
    accepted = 0
    tic = time.monotonic()
    for i in range(2001, int(indices[-1])+1):
        previous_success = tracker.filter.last_success_mahalanobis
        did_update = tracker.on_imu_measurement(i*5000, gyro[i, :, None], accel[i, :, None])
        if did_update:
            updates += 1
            accepted += tracker.filter.last_success_mahalanobis != previous_success
        if i == indices[cursor]:
            state = tracker.filter.state
            row = np.r_[state.s_p.ravel(), state.s_v.ravel()]
            cov_diag = np.diag(tracker.filter.Sigma15)
            if not np.isfinite(row).all() or not np.isfinite(cov_diag).all():
                raise FloatingPointError(f"Nonfinite EKF state at {i/200:.3f}s")
            saved.append(row)
            q = Rotation.from_matrix(state.s_R).as_quat()
            debug.append(np.r_[q[[3, 0, 1, 2]], state.s_bg.ravel(), state.s_ba.ravel(), cov_diag])
            cursor += 1
            if cursor == len(indices):
                break
        if i % 12000 == 0:
            print(f"    IMU {i/200:.0f}/{indices[-1]/200:.0f} s, wall {time.monotonic()-tic:.1f} s", flush=True)
    return np.asarray(saved), np.asarray(debug), dict(
        network_updates=updates, mahalanobis_successes=accepted,
        elapsed_seconds=time.monotonic()-tic,
        final_covariance_min_eigenvalue=float(np.linalg.eigvalsh(tracker.filter.Sigma).min()),
    )


def run_record(folder, data_root, out_root, max_seconds, ckpt, param, cpu, manifest):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.getLogger().setLevel(logging.ERROR)
    session, person = folder.parts[-4], folder.parts[-2]
    print(f'{session}/{person}: calibrated IMU -> EqNIO-TLIO EKF', flush=True)
    hashes = {name:sha256(folder/name) for name in SENSORS.values()}
    ns, gyro, accel, initial_r, indices, quality = prepare_body(folder, max_seconds)
    out = out_root/session/person/SUBDIR
    out.mkdir(parents=True)
    write_json(out.parent/'input_quality.json', quality)
    states, debug, status = infer(ns, gyro, accel, initial_r, indices, ckpt, param, cpu)
    pred_ns = ns[indices]
    pd.DataFrame(dict(timestamp_ns=pred_ns, local_x_m=states[:,0], local_y_m=states[:,1],
                      local_z_m=states[:,2], local_vx_mps=states[:,3], local_vy_mps=states[:,4],
                      local_vz_mps=states[:,5])).to_csv(out/'trajectory_local.csv', index=False)
    np.savez_compressed(out/'filter_diagnostics.npz', timestamp_ns=pred_ns,
                        quat_wxyz=debug[:,:4], gyro_bias=debug[:,4:7], accel_bias=debug[:,7:10],
                        state_covariance_diagonal=debug[:,10:])
    write_uncertainty_csv(out/'uncertainty_local.csv', pred_ns, debug[:,10:])
    # First reference read occurs after all local outputs have been saved.
    reference=data_root/'recordings'/session/'reference'/person/'trajectory-gt.csv'
    reference_hash=sha256(reference)
    metrics=evaluate(reference,pred_ns,states[:,:2],states[:,3:5],out,'TLIO O2 + EKF')
    assert all(sha256(folder/name)==digest for name,digest in hashes.items())
    assert sha256(reference)==reference_hash
    summary=dict(session=session,person=person,variant='o2',method=METHOD,
                 checkpoint_sha256=manifest['checkpoint_sha256'],input_sha256=hashes,
                 reference_sha256=reference_hash,inputs_unchanged=True,metrics=metrics,
                 predictions=len(indices),quality=quality,filter_status=status)
    write_json(out/'summary.json',summary)
    print(f"  {session}/{person}: saved {len(indices)} states; offline diagnostic RMSE {metrics['horizontal_rmse_m']:.2f} m",flush=True)
    return dict(session=session,person=person,variant='o2',**status,
                **{k:v for k,v in metrics.items() if isinstance(v,(int,float))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/maxson/data/PedROAM'))
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT.parent/'outputs/eqnio-pretrained/TLIO_o2')
    parser.add_argument('--out-root', type=Path, default=ROOT/'outputs/tlio-current/inference')
    parser.add_argument('--session')
    parser.add_argument('--person')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.out_root.resolve().is_relative_to(args.data_root.resolve()):
        parser.error('Outputs must be outside the dataset')
    ckpt, param = args.checkpoint_root/'checkpoint_best.pt', args.checkpoint_root/'parameters.json'
    config = json.loads(param.read_text())
    assert config['arch'] == ARCH and config['imu_freq'] == 200.
    assert config['window_time'] == 1. and config['past_time'] == 0. and config['future_time'] == 0.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.getLogger().setLevel(logging.WARNING)
    selected = [f for f in sorted((args.data_root/'recordings').glob('*/phones/*/sensorlogger'))
                if (not args.session or f.parts[-4] == args.session) and (not args.person or f.parts[-2] == args.person)]
    if not selected:
        raise ValueError('No selected recordings')
    args.out_root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        workers=args.workers, device='cpu' if args.cpu else 'cuda',
        method=METHOD, model_family='tlio', trajectory_subdir=SUBDIR, variant='o2',
        bias_estimator='ekf', input_files=list(SENSORS.values()),
        checkpoint_sha256=sha256(ckpt), parameters_sha256=sha256(param), architecture=ARCH,
        script_sha256=sha256(__file__), filter_config=FILTER_CONFIG,
        versions={p: importlib.metadata.version(p) for p in ['torch','numpy','scipy','numba','ahrs']},
        initialization='Madgwick beta=.033 at 10 s; zero position, velocity and residual biases; upstream first-update p/v reset',
        attitude='Madgwick only for initialization, then original TLIO EKF and within-window gyro integration',
        network_hz=200, filter_hz=200, update_hz=20, window_samples=200,
        evaluation='GT opened after inference only; shared cumulative 10 m fixed yaw/translation; XY metrics',
        original_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [
            ROOT/'TLIO-master/src/tracker/imu_tracker.py', ROOT/'TLIO-master/src/tracker/scekf.py',
            ROOT/'TLIO-master/src/tracker/meas_source_torchscript.py',
        ]},
    )
    write_json(args.out_root/'run.json', manifest)
    rows, failures = [], []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        jobs = {pool.submit(run_record, folder, args.data_root, args.out_root, args.max_seconds,
                            ckpt, param, args.cpu, manifest): folder for folder in selected}
        for future in as_completed(jobs):
            folder = jobs[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append(dict(session=folder.parts[-4],person=folder.parts[-2],error=repr(exc)))
            pd.DataFrame(rows).sort_values(['session','person']).to_csv(args.out_root/'summary.csv',index=False) if rows else None
            write_json(args.out_root/'failures.json',failures)
    if failures:
        raise SystemExit(f'{len(failures)} failed recordings: {failures}')


if __name__=='__main__':
    main()
