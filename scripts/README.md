# PedROAM: Madgwick + RoNIN O2

The maintained pipeline uses calibrated `Gyroscope.csv` and
`TotalAcceleration.csv`, Madgwick 6D with gain **0.033**, and the released
EqNIO RoNIN O2 checkpoint. It does not read uncalibrated sensors, magnetic
streams, or phone orientation. No additional gyro bias is subtracted.

## Files

- `attitude_6d.py`: compiled Madgwick IMU update, checked against AHRS 0.4.0.
- `ronin_pedroam_6d.py`: sensor preparation, frozen O2 inference, local trajectory
  integration, and an offline alignment diagnostic.
- `evaluate_initial_alignment.py`: the selected initial-position and cumulative
  10-metre heading correction, trajectory plots, and final metrics.
- `summarize_ronin_6d.py`: independently verifies intermediate trajectories using
  velocity reintegration and SVD, then summarizes the diagnostic metrics.
- `test_*.py`: numerical tests for attitude, sensor selection, and alignment.

## Run

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
python -B scripts/summarize_ronin_6d.py "$run_root/inference"
python -B scripts/evaluate_initial_alignment.py \
  --result-root "$run_root/inference" \
  --out-root "$run_root/alignment-10m" \
  --initialization-distance-m 10
```

With no path arguments, the pipeline uses `outputs/ronin-current/`.
Existing inference/alignment directories are never overwritten. To run a
single recording, add `--session 0421-AM-2026 --person P01` to inference.
`--max-seconds 120` provides a short test; `--cpu` disables CUDA.

## Processing and alignment

The pipeline interpolates IMU to 100 Hz, computes forward Madgwick attitude,
then interpolates IMU and attitude to 200 Hz. It preserves gravity and rotates
both sensor vectors into one continuous local frame. The first 10 seconds
initialize the filter. O2 consumes 200-sample windows at stride 10; predicted
XY velocities are assigned to window midpoints and integrated at 20 Hz.

Ground truth enters only after inference. Starting at the first overlapping
valid GT position, accumulate successive horizontal displacement lengths.
Stop at the first sample reaching **10 metres**. Predicted and reference
trajectories use this same initial interval. Subtract their accumulated
vector directions to obtain one yaw offset, then translate the initial
predicted position to the initial GT position. Keep this transform fixed.
There is no scale fit, straightness selection, or later heading update.

The direction comes from accumulated position differences, not measured phone
yaw. Initialization ends at the threshold-crossing timestamp; earlier points
are transformed retrospectively for plotting. Primary metrics exclude this
initialization period. GT jumps and gaps are not silently filtered: the actual
initialization distance, net displacement, duration, and maximum gap are saved.

## Current results

`outputs/ronin-current/` contains only the selected configuration:

- `alignment-10m/aggregate.csv`: final mean and median sequence RMSE.
- `alignment-10m/per_sequence.csv`: individual recording metrics.
- `alignment-10m/Madgwick/<session>/<person>/`: corrected trajectory CSV,
  evaluation CSV, summary JSON, and `trajectory.png`.
- `inference/`: original local predictions, attitude arrays, consumed-input
  hashes, checkpoint metadata, and independent verification.
- `cleanup_verification.json`: comparison against the selected pre-cleanup
  results, confirming cleanup did not change the algorithm's predictions.

The whole-trajectory best-fit metrics under `inference/` are diagnostics;
they are not the selected initial-only evaluation. Generated files remain
ignored by Git. The original upstream RONIN and TLIO source trees are separate
from these maintained inference scripts.
