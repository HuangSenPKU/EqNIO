#!/usr/bin/env python3
"""Audit matched RoNIN/TLIO evaluations and write a dataset-paper baseline report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ronin_pedroam_6d import ROOT, sha256, write_json
from evaluate_initial_alignment import relative_rmse


def summarize(ronin, tlio):
    report=tlio/'report'
    report.mkdir(exist_ok=False)
    roots={'RoNIN':ronin,'TLIO':tlio}
    tables={name:pd.read_csv(root/'alignment-10m/per_sequence.csv').set_index(['session','person']).sort_index()
            for name,root in roots.items()}
    assert tables['RoNIN'].index.equals(tables['TLIO'].index)
    assert len(tables['TLIO'])==13, 'Report requires all 13 recordings'
    audits=[]
    for key in tables['TLIO'].index:
        evaluations={}
        summaries={}
        local_times={}
        for name,root in roots.items():
            table=tables[name]
            row=table.loc[key]
            aligned_dir=root/'alignment-10m'/row.estimator/key[0]/key[1]
            ev=pd.read_csv(aligned_dir/'evaluation.csv')
            provenance=json.loads((aligned_dir/'summary.json').read_text())
            source=Path(provenance['source'])
            local=pd.read_csv(source/'trajectory_local.csv')
            summary=json.loads((source/'summary.json').read_text())
            assert summary['inputs_unchanged']
            assert sha256(source/'trajectory_local.csv')==provenance['source_trajectory_sha256']
            values=ev[['initial_aligned_e_m','initial_aligned_n_m']].to_numpy()
            ref=ev[['gt_e_m','gt_n_m']].to_numpy()
            ns=ev.timestamp_ns.to_numpy(np.int64)
            t=(ns-ns[0])*1e-9
            post=ns>row.initialization_end_timestamp_ns
            assert post.sum()==row.post_initialization_samples
            error=values-ref
            independent_rmse=float(np.sqrt(np.mean(np.sum(error[post]**2,axis=1))))
            np.testing.assert_allclose(independent_rmse,row.initial_post_rmse_m,atol=1e-8)
            np.testing.assert_allclose(relative_rmse(t,error,row.initialization_end_s),row.initial_post_relative60_rmse_m,atol=1e-8)
            rotation=np.array(provenance['rotation'])
            translation=np.array(provenance['translation'])
            pred_time=(local.timestamp_ns.to_numpy(np.int64)-ns[0])*1e-9
            pred=np.column_stack([np.interp(t,pred_time,local[c]) for c in ['local_x_m','local_y_m']])
            np.testing.assert_allclose(pred@rotation.T+translation,values,atol=1e-8)
            j=np.flatnonzero(np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))]>=10)[0]
            dp,dg=pred[j]-pred[0],ref[j]-ref[0]
            angle=np.angle(complex(*dg)/complex(*dp))
            independent_rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            np.testing.assert_allclose(rotation,independent_rotation,atol=1e-9)
            np.testing.assert_allclose(values[0],ref[0],atol=1e-9)
            assert ns[j]==row.initialization_end_timestamp_ns
            evaluations[name]=ev
            summaries[name]=summary
            local_times[name]=local.timestamp_ns.to_numpy(np.int64)
        np.testing.assert_array_equal(local_times['RoNIN'],local_times['TLIO'])
        columns=['timestamp_ns','gt_e_m','gt_n_m','post_initialization']
        pd.testing.assert_frame_equal(evaluations['RoNIN'][columns],evaluations['TLIO'][columns],check_exact=True)
        for field in ['input_sha256','reference_sha256']:
            assert summaries['RoNIN'][field]==summaries['TLIO'][field]
        audits.append(dict(session=key[0],person=key[1],same_input_hashes=True,same_reference_hash=True,
                           identical_prediction_timestamps=True,identical_evaluation_samples=True,
                           identical_initialization_interval=True,independent_alignment_and_metric_check=True))
    metrics=['initial_post_rmse_m','initial_post_relative60_rmse_m']
    assert all(np.isfinite(table[metrics].to_numpy()).all() for table in tables.values())
    comparison=pd.DataFrame(index=tables['TLIO'].index)
    for name,table in tables.items():
        comparison[name.lower()+'_post_rmse_m']=table.initial_post_rmse_m
        comparison[name.lower()+'_relative60_rmse_m']=table.initial_post_relative60_rmse_m
    comparison.reset_index().to_csv(report/'baseline_comparison_per_sequence.csv',index=False)
    rows=[]
    for name,table in tables.items():
        rows.append(dict(baseline=name,sequences=len(table),mean_post_rmse_m=table.initial_post_rmse_m.mean(),
                         median_post_rmse_m=table.initial_post_rmse_m.median(),
                         mean_relative60_rmse_m=table.initial_post_relative60_rmse_m.mean()))
    aggregate=pd.DataFrame(rows).set_index('baseline')
    aggregate.to_csv(report/'baseline_comparison_aggregate.csv')
    write_json(report/'comparison_verification.json',dict(recordings=13,checks=audits,
               script_sha256=sha256(__file__),all_checks_passed=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    y=np.arange(len(comparison)); labels=[f'{session} / {person}' for session,person in comparison.index]
    for ax,metric,title in zip(axes,metrics,['Post-initialization position RMSE','60 s relative displacement RMSE']):
        ax.barh(y-.18,tables['RoNIN'][metric],height=.35,label='EqNIO-RoNIN O2',color='#3678A5')
        ax.barh(y+.18,tables['TLIO'][metric],height=.35,label='EqNIO-TLIO O2 + EKF',color='#C27C35')
        ax.set_yticks(y,labels if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set(xlabel='RMSE (m)',title=title)
        ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.savefig(report/'baseline_comparison.png',dpi=180)
    plt.close(fig)
    r,t=aggregate.loc['RoNIN'],aggregate.loc['TLIO']
    metric_rows='\n'.join(f'| {label} | {r[key]:.2f} | {t[key]:.2f} |' for label,key in [
        ('Mean position RMSE (m)','mean_post_rmse_m'),('Median position RMSE (m)','median_post_rmse_m'),
        ('Mean 60 s relative displacement RMSE (m)','mean_relative60_rmse_m')])
    worse=int((comparison.tlio_post_rmse_m>comparison.ronin_post_rmse_m).sum())
    worst=comparison.tlio_post_rmse_m.idxmax()
    worst_value=float(comparison.loc[worst,'tlio_post_rmse_m'])
    sequence_rows='\n'.join(f'| {s} / {p} | {v.ronin_post_rmse_m:.2f} | {v.tlio_post_rmse_m:.2f} |'
                            for (s,p),v in comparison.iterrows())
    text=f'''# EqNIO-TLIO Baseline Evaluation on PedROAM

## Purpose and baseline configuration

We evaluate the released [EqNIO](https://arxiv.org/abs/2408.06321)-TLIO O2 model with the [TLIO](https://cathias.github.io/TLIO/) stochastic-cloning extended Kalman filter (EKF) as a pretrained baseline for the PedROAM dataset. Weights are frozen, and no training or fine-tuning is performed on PedROAM. All 13 recordings are retained. This experiment evaluates transfer of an existing method, rather than introducing a new odometry algorithm.

The model uses architecture `eq_o2_frame_fullCov_2vec_2deep` and checkpoint `TLIO_o2/checkpoint_best.pt`. Inputs are calibrated `Gyroscope.csv` and `TotalAcceleration.csv`, in rad/s and m/s². Magnetometer data, uncalibrated gyroscope streams, and phone-provided orientations are excluded. IMU streams are synchronized and linearly resampled to 100 Hz, then interpolated to 200 Hz, matching the RoNIN baseline's sensor preparation.

## Filter initialization and inference

Madgwick IMU filtering with gain 0.033 supplies the attitude after a 10 s warm-up. Initial position, velocity, and residual accelerometer/gyroscope biases are zero. The upstream non-VIO first-update reset of position and velocity is retained. No reference attitude, velocity, position, or bias is passed to the filter.

After initialization, the original TLIO EKF estimates attitude, velocity, position, and residual biases. The network consumes 200-sample windows in a local gravity-aligned frame whose yaw is anchored to the window's initial cloned state; within-window rotation is obtained by gyro integration. Every 50 ms, predicted three-dimensional displacement and covariance constrain the EKF. The learned-frame inverse rotation and covariance transformation are preserved. We retain the upstream command-line filter defaults, including gravity 9.81 m/s², measurement covariance scale 10, and Mahalanobis gating. Acceleration retains gravity at the network input.

This is the full network-plus-EKF baseline. Unlike RoNIN, it does not integrate network predictions as instantaneous velocities, and it uses Madgwick only for initial attitude rather than continuously.

## Shared evaluation protocol

Both baselines use identical prediction timestamps, valid reference samples, and initialization intervals for every recording. Ground truth is opened only after local inference. Starting at the first overlapping valid reference position, we accumulate horizontal reference path length until it first reaches 10 m. Over that same interval, the direction difference between reference and predicted net displacements defines one fixed yaw offset. A translation anchors the initial predicted position to the reference. No scale is fitted and no later reference-based correction is applied.

The reported direction is inferred from position changes, not measured device yaw. The transform becomes available only after the initialization interval. Primary metrics exclude this interval and evaluate horizontal position only; the EKF itself remains three-dimensional. The 60 s relative displacement metric uses the same fixed transform, with a 0.1 s endpoint tolerance and no pairs spanning reference gaps longer than 2 s. Metrics are computed per recording and averaged with equal recording weights.

## Results

| Metric | EqNIO-RoNIN O2 | EqNIO-TLIO O2 + EKF |
| --- | ---: | ---: |
{metric_rows}

| Recording | RoNIN position RMSE (m) | TLIO position RMSE (m) |
| --- | ---: | ---: |
{sequence_rows}

Under this configuration, TLIO has higher position RMSE on {worse} of the 13 recordings. The largest error occurs on `{worst[0]}/{worst[1]}` ({worst_value:.2f} m); all recordings remain in the aggregate.

The comparison reflects the complete baseline pipelines under a shared initialization and scoring protocol. The checkpoints originate from their respective source training datasets; the comparison does not isolate the contribution of training data, backbone, attitude estimator, or filter. Whole-trajectory best-fit alignment is retained only as a diagnostic and is excluded from the table above.

## Limitations and reproducibility

Initial reference inconsistencies in `0422-AM1-2026/P02` and `P04` remain included: the 10 m threshold is crossed by a roughly 37 m reference jump near the start. Initialization connects valid reference points across gaps without smoothing or silent rejection. Fixed alignment cannot eliminate subsequent drift, and zero-velocity initialization need not match the true motion state. The prior RoNIN configuration was selected through exploration on these recordings; this TLIO run retains upstream filter defaults without tuning them against these results.

Results and plots are available in [the alignment folder](../alignment-10m/), with [per-recording comparison data](baseline_comparison_per_sequence.csv), [aggregate data](baseline_comparison_aggregate.csv), and [matched-protocol verification](comparison_verification.json). The [run manifest](../inference/run.json) records checkpoint hashes, filter settings, and software versions. [Execution instructions](../../../scripts/README.md) describe reproduction. Original dataset files and pretrained weights are unchanged.

## Implementation validation

All 12 adapter and shared-protocol tests passed. Separate checks verified learned displacement/covariance rotation behavior and positive covariance eigenvalues on sampled windows. The 120 s CPU/GPU smoke trajectories differed by at most 0.022 m horizontally. The upstream suite reported 6 failures out of 17 tests: five legacy tensor-shape assertions and one rotation-logarithm branch comparison. Explicit-shape covariance checks and a 100-case rotation-matrix audit passed; the failed rotation helper is not used in the inference loop. Upstream model and filter code were left unchanged. Details are retained in [validation logs](../validation/) and the [network verification record](../verification_network.json).
'''
    (report/'tlio_baseline_evaluation_en.md').write_text(text)
    print(aggregate.to_string())
    print(f'All 13 matched-protocol checks passed: {report}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ronin-root',type=Path,default=ROOT/'outputs/ronin-current')
    parser.add_argument('--tlio-root',type=Path,default=ROOT/'outputs/tlio-current')
    args=parser.parse_args()
    summarize(args.ronin_root,args.tlio_root)
