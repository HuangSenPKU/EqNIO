# Repository Guidelines

## Project Structure & Module Organization

- `TLIO-master/src/`: filter-based odometry, with models in `network/`, input pipelines in `dataloader/`, EKF code in `tracker/`, and evaluation scripts in `analysis/`.
- `RONIN/source/`: end-to-end odometry models, preprocessing, baselines, and trajectory metrics. Dataset splits live in `RONIN/lists/`.
- TLIO unit tests reside in `network/tests/` and `tracker/tests/`; `images/` contains documentation assets.
- `environment.yml` defines the Conda environment; `README.md` provides dataset, checkpoint, and experiment instructions.

## Build, Test, and Development Commands

Run from the repository root unless noted. No separate build step is configured.

- `conda env create -f environment.yml`, then `conda activate eqnio`: install and activate dependencies. Use `.yml`; the README's `.yaml` filename is incorrect.
- `python TLIO-master/src/main_net.py --help`: inspect neural-network training and evaluation options. Use `--mode train` or `--mode test`, with explicit dataset, architecture, output, and checkpoint arguments.
- `python TLIO-master/src/main_filter.py --help`: inspect EKF inference options.
- `python RONIN/source/ronin_resnet.py --help`: inspect RoNIN options; root-relative split paths start with `RONIN/lists/`.
- From `TLIO-master/src/`, run `python -m unittest tracker.tests.test_state network.tests.test_covariance_parametrization` for core unit tests.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Preserve existing public APIs and architecture identifiers. Follow neighboring formatting; no formatter or linter configuration is tracked. Document tensor shapes, coordinate frames, timestamp units, and angle conventions when changing numerical interfaces.

## Testing Guidelines

Core tests use `unittest` with NumPy and PyTorch assertions. Name files `test_*.py` and methods `test_*`; add regression checks for numerical changes. No coverage threshold is configured. Broad discovery encounters `test_eq_VNtransformer.py`, which imports an absent model module. For model or filter changes, report representative trajectory metrics alongside unit-test results.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as “Update download links for pretrained models”; no strict prefix convention is evident. Keep commits focused. PRs should explain behavior changes, link relevant issues, list validation commands and dataset/checkpoint settings, and include trajectory plots when useful.

## Data & Outputs

Treat `/home/maxson/data/PedROAM` as input data; inspect write paths before running experiments. Save temporary results under repository-local `outputs/` and official results under `results/<session-name>/<pedestrian-number>/<test-name>/`. Keep datasets, checkpoints, and generated outputs out of commits.
