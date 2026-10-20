#!/usr/bin/env bash
# Run one 180-second recording through inference and initial-only evaluation.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_baseline_example.sh [ronin|tlio|both] [NEW_OUTPUT_DIRECTORY]

Defaults: both baselines, recording 0423-PM-2026/P02, first 180 seconds.
The output directory must not exist. A unique directory is generated if omitted.
Activate the eqnio environment first, or set PYTHON_BIN to its Python executable.
Optional environment variables: DATA_ROOT, CHECKPOINT_ROOT, CPU=1.
EOF
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# > 2 )); then usage >&2; exit 2; fi
model=${1:-both}
case "$model" in ronin|tlio|both) ;; *) usage >&2; exit 2 ;; esac
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/home/maxson/data/PedROAM}
checkpoint_root=${CHECKPOINT_ROOT:-../outputs/eqnio-pretrained}
example_root=${2:-outputs/baseline-example-$(date +%Y%m%d-%H%M%S)-$$}
cpu_args=()
if [[ ${CPU:-0} == 1 ]]; then cpu_args=(--cpu); fi

# Check paths before creating anything; never write into the input dataset.
"$python_bin" -B - "$example_root" "$data_root" "$checkpoint_root" "$model" <<'PY'
from pathlib import Path
import sys
import torch, numpy, pandas, scipy, ahrs
out, data, checkpoints = map(lambda x: Path(x).resolve(), sys.argv[1:4])
model = sys.argv[4]
if out.exists():
    raise SystemExit(f'Output already exists; choose a new path: {out}')
if out.is_relative_to(data):
    raise SystemExit('Output must be outside the input dataset')
folder = data/'recordings/0423-PM-2026/phones/P02/sensorlogger'
required = [folder/'Gyroscope.csv', folder/'TotalAcceleration.csv',
            data/'recordings/0423-PM-2026/reference/P02/trajectory-gt.csv']
if model in ('ronin', 'both'):
    required += [checkpoints/'Ronin_o2/checkpoint_38.pt', checkpoints/'Ronin_o2/config.json']
if model in ('tlio', 'both'):
    required += [checkpoints/'TLIO_o2/checkpoint_best.pt', checkpoints/'TLIO_o2/parameters.json']
for path in required:
    if not path.is_file():
        raise SystemExit(f'Missing input: {path}')
print(f'Python: {sys.executable}; PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}')
PY
mkdir -p -- "$(dirname -- "$example_root")"
mkdir -- "$example_root"
example_root=$(cd -- "$example_root" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$repo_root/outputs/.numba"

if [[ $model == ronin || $model == both ]]; then
  "$python_bin" -B scripts/ronin_pedroam_6d.py \
    --data-root "$data_root" --checkpoint-root "$checkpoint_root" \
    --session 0423-PM-2026 --person P02 --max-seconds 180 \
    --out-root "$example_root/ronin/inference" "${cpu_args[@]}"
  "$python_bin" -B scripts/evaluate_initial_alignment.py \
    --result-root "$example_root/ronin/inference" \
    --out-root "$example_root/ronin/alignment-10m-continuous" \
    --initialization-distance-m 10 --label Madgwick
fi
if [[ $model == tlio || $model == both ]]; then
  "$python_bin" -B scripts/tlio_pedroam.py \
    --data-root "$data_root" --checkpoint-root "$checkpoint_root/TLIO_o2" \
    --session 0423-PM-2026 --person P02 --max-seconds 180 --workers 1 \
    --out-root "$example_root/tlio/inference" "${cpu_args[@]}"
  "$python_bin" -B scripts/evaluate_initial_alignment.py \
    --result-root "$example_root/tlio/inference" \
    --out-root "$example_root/tlio/alignment-10m-continuous" \
    --initialization-distance-m 10 --label TLIO-EKF
fi
"$python_bin" -B - "$example_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd
root=Path(sys.argv[1])
for baseline in ('ronin','tlio'):
    path=root/baseline/'alignment-10m-continuous/aggregate.csv'
    if path.exists():
        print(f'\n{baseline.upper()} — 180-second example, not the full dataset result')
        print(pd.read_csv(path).to_string(index=False))
print(f'\nCompleted. Outputs: {root}')
PY
