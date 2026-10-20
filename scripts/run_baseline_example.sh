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

# The example is a preset of the general-purpose runner.
exec "$python_bin" -B scripts/run_baselines.py \
  --data-root "$data_root" --checkpoint-root "$checkpoint_root" \
  --out-root "$example_root" --model "$model" \
  --session 0423-PM-2026 --person P02 --max-seconds 180 --workers 1 \
  "${cpu_args[@]}"
