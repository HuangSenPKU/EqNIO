# PedROAM Pretrained Inertial Odometry Baselines

中文流程、输入输出说明及可执行样例见 [Baseline 使用指南](../docs/BASELINE_USAGE_ZH.md)。快速运行：`bash scripts/run_baseline_example.sh both`。

Run all selected recordings and archive results using the PedROAM convention:

```bash
conda activate eqnio
python -B scripts/run_baselines.py \
  --data-root /home/maxson/data/PedROAM --test-name eqnio-baselines-v1
```

The runner defaults to both baselines. Use `--model ronin|tlio|both`,
`--session`, `--person`, `--max-seconds`, `--workers`, or `--cpu` as needed.
The data root must contain `recordings/`. `--checkpoint-root` points to the
parent of `Ronin_o2/` and `TLIO_o2/`. Relative paths resolve from the caller's
working directory. `--dry-run` checks paths and prints commands without running.

Both inference scripts export initial-aligned trajectories before evaluation.
The runner forwards `--initialization-distance-m` (default 10) to both models
and reports, storing reports under `alignment-<distance>m-continuous/`.

The runner defaults to `<data-root>/results/<session>/<person>/<test-name>/`,
with `ronin/` and/or `tlio/` inside each experiment. Each model folder contains
its trajectory CSVs, comparison plot, inference metadata and an `evaluation/`
folder for initial-alignment metrics. The experiment's `summary.csv` contains
that recording's per-model metrics, alongside `run.json` and `reference-audit/`.
Existing experiments are refused; existing results bases and other experiments
are preserved. Omitting `--test-name` generates a unique timestamped name.

`--out-root` now means the results **base**, not the old batch work directory.
`--work-root` defaults to repository `outputs/baseline-work/<test-name>/<timestamp>/`; it
retains caches, logs, the old per-model inference/report layout, and batch-wide
`summary.csv`/`per_sequence.csv`. It must be a fresh directory outside the data
and checkpoints. Only completed runs are archived. Within PedROAM, writes are
restricted to `results/`; recordings, rosbags and environment inputs are protected.
The 180-second `run_baseline_example.sh` uses the same archive convention and
accepts an optional results base as its second argument, plus `TEST_NAME`.

The RoNIN pipeline uses calibrated `Gyroscope.csv` and
`TotalAcceleration.csv`, Madgwick 6D with gain **0.033**, and the released
EqNIO RoNIN O2 checkpoint. It does not read uncalibrated sensors, magnetic
streams, or phone orientation. No additional gyro bias is subtracted.

## RoNIN files

- `attitude_6d.py`: compiled Madgwick IMU update, checked against AHRS 0.4.0.
- `ronin_pedroam_6d.py`: sensor preparation, frozen O2 inference, local trajectory
  integration, automatic initial position/heading alignment, and an offline alignment diagnostic.
- `trajectory_alignment.py`: shared initial rigid XY transform.
- `evaluate_initial_alignment.py`: reuses the saved initial transform for
  trajectory plots and final metrics; also supports TLIO and older inference outputs.
- `reference_quality.py`: shared causal reference-continuity screening.
- `audit_reference.py`: reference jumps/gaps, raw matching metadata, and plots.
- `summarize_ronin_6d.py`: independently verifies intermediate trajectories using
  velocity reintegration and SVD, then summarizes the diagnostic metrics.
- `test_*.py`: numerical tests for attitude, sensor selection, and alignment.

## Run RoNIN

Use the repository's `eqnio` Conda environment and install the extra dependency:

```bash
python -m pip install -r scripts/requirements-inference.txt
python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

The dataset defaults to `/home/maxson/data/PedROAM`. The checkpoint defaults to
`../outputs/eqnio-pretrained/Ronin_o2/checkpoint_38.pt`, with its adjacent
`config.json`. Inputs are read-only and outputs must be outside the dataset.

Run all 13 available recordings using a fresh output directory:

```bash
run_root=outputs/ronin-new-run
python -B scripts/ronin_pedroam_6d.py --out-root "$run_root/inference"
# The inference command already writes trajectory_initial_aligned.csv beside
# trajectory_local.csv in each recording directory. Optional reports follow:
python -B scripts/summarize_ronin_6d.py "$run_root/inference"
python -B scripts/evaluate_initial_alignment.py \
  --result-root "$run_root/inference" \
  --out-root "$run_root/alignment-10m-continuous" \
  --initialization-distance-m 10
```

With no path arguments, the pipeline uses `outputs/ronin-current/`.
Existing inference/alignment directories are never overwritten. To run a
single recording, add `--session 0421-AM-2026 --person P01` to inference.
`--max-seconds 120` provides a short test; `--cpu` disables CUDA.

RoNIN also writes `initial_alignment.json` with rotation, translation, initialization
timestamps and source hashes before scoring. `--initialization-distance-m` defaults
to 10; use the same distance in optional evaluation to reuse the saved transform.
Insufficient continuous reference travel or undefined heading fails explicitly,
retaining local outputs. Aligned CSV timestamps stay at 20 Hz, with rotated ENU
positions/velocities and an `initialization_complete` flag. Earlier points use
the retrospectively applied transform.

## Processing and alignment

The pipeline interpolates IMU to 100 Hz, computes forward Madgwick attitude,
then interpolates IMU and attitude to 200 Hz. It preserves gravity and rotates
both sensor vectors into one continuous local frame. The first 10 seconds
initialize the filter. O2 consumes 200-sample windows at stride 10; predicted
XY velocities are assigned to window midpoints and integrated at 20 Hz.

Ground truth enters only after inference. For these approximately 1 Hz pedestrian
references, a gap above **1.5 s** or horizontal step speed above **3 m/s** resets
initialization. Wait **10 s** after the last flagged transition, select a new
position anchor, then accumulate **10 metres** within that continuous segment.
Without a discontinuity, retain the original first-sample anchor. Both models
use the same GT-selected interval. Its net-displacement directions define one
yaw offset; translation anchors the prediction at the **selected start**.
The transform stays fixed. No scale, straightness selection, or later updates
are used. A net displacement below **0.5 m** in either trajectory fails explicitly
without searching for another model-dependent interval.

The direction comes from accumulated position differences, not measured phone
yaw. Initialization ends at the threshold-crossing timestamp; earlier points
are transformed retrospectively for plotting. Primary metrics exclude this
initialization period. Raw coordinates and all later position-scoring samples
are retained. Reference continuity flags are exported; 60 s relative-displacement
pairs crossing a flagged transition are excluded. Screening cannot repair a
reference-frame reset or certify GT accuracy. Thresholds are CLI options, and
`--protocol legacy` reproduces the previous gap-connecting initialization.

The [reference-jump investigation](../docs/REFERENCE_JUMP_FIX_ZH.md) records the
evidence, revised protocol, results, and commands to reevaluate saved predictions.
Old `alignment-10m/` results remain available as legacy results.

## Current RoNIN results

`outputs/ronin-current/` contains only the selected configuration:

- `alignment-10m-continuous/aggregate.csv`: final mean and median sequence RMSE.
- `alignment-10m-continuous/per_sequence.csv`: individual recording metrics.
- `alignment-10m-continuous/Madgwick/<session>/<person>/`: corrected trajectory CSV,
  evaluation CSV, summary JSON, and `trajectory.png`.
- `inference/`: original local predictions, attitude arrays, consumed-input
  hashes, checkpoint metadata, and independent verification.
- `cleanup_verification.json`: comparison against the selected pre-cleanup
  results, confirming cleanup did not change the algorithm's predictions.

The whole-trajectory best-fit metrics under `inference/` are diagnostics;
they are not the selected initial-only evaluation. Generated files remain
ignored by Git. The original upstream RONIN and TLIO source trees are separate
from these maintained inference scripts.

## EqNIO-TLIO O2 with the original EKF

`tlio_pedroam.py` runs the released `TLIO_o2/checkpoint_best.pt`
(`eq_o2_frame_fullCov_2vec_2deep`) with the upstream stochastic-cloning EKF.
It reads the same two calibrated IMU streams and uses the same 100-to-200 Hz
interpolation and output timestamp grid as the RoNIN baseline.

Madgwick (gain 0.033) supplies the attitude at 10 s only. The EKF starts with
zero position, velocity, and residual biases; the upstream non-VIO first-update
position/velocity reset is retained. Thereafter, TLIO estimates attitude and
residual biases internally. Its network receives gyro-integrated, locally
heading-aligned one-second windows and predicts displacement plus covariance.
The upstream learned-frame inverse rotation, covariance transformation, noise
settings, covariance scale of 10, and Mahalanobis gating remain unchanged.
No ground-truth attitude, velocity, position, or bias enters the filter.

```bash
run_root=outputs/tlio-new-run
python -B scripts/tlio_pedroam.py --workers 8 \
  --out-root "$run_root/inference"
# Initial-aligned CSV and comparison plot are already available. Optional report:
python -B scripts/evaluate_initial_alignment.py \
  --result-root "$run_root/inference" \
  --out-root "$run_root/alignment-10m-continuous" --label TLIO-EKF
```

Use `--session`, `--person`, and `--max-seconds 120` for a short run.
Each recording runs in a separate process; `--workers` controls concurrency.
The default output is `outputs/tlio-current/inference`, and existing run
folders are never overwritten. Dataset and checkpoint files remain read-only.
The current run stores local trajectories, compressed filter diagnostics,
input/checkpoint hashes, and explicit filter configuration. Its final aligned
trajectories use the same cumulative 10 m reference-assisted initialization as
RoNIN and are exported directly as `trajectory_initial_aligned.csv`, with
`initial_alignment.json`. `trajectory.png` separates raw local, initial-aligned,
and full-overlap fitted trajectories, with matched-time error curves. The runner's
report reuses the transform and the same plot style. Primary metrics exclude the
initialization interval. `uncertainty_local.csv` remains in the local filter frame;
only XY position and velocity are transformed into ENU.

This is the **full TLIO network-plus-EKF baseline**, not standalone integration
of the network's displacement predictions. RoNIN uses Madgwick throughout,
whereas TLIO uses it only to initialize its own attitude estimator. Ground-truth
position alignment is applied after inference to both baselines.

After all 13 TLIO recordings and initial-alignment evaluations finish, generate
an audited comparison and the English dataset-paper evaluation draft:

```bash
python -B scripts/summarize_tlio_baseline.py \
  --ronin-root outputs/ronin-current --tlio-root outputs/tlio-current
```

The report verifies identical IMU/reference hashes, prediction timestamps,
scoring samples, and 10 m initialization intervals. It independently checks the
fixed transforms and metrics before writing comparison CSVs and a plot to
`outputs/tlio-current/report-continuous/`. Use `--alignment-subdir alignment-10m
--report-subdir report-legacy-new` to audit legacy outputs into a fresh folder.
