# X-ray Attention Monitoring

## Current research: v5 fatigue gaze-factor protocol

The active entry point is `scripts/run_efa_v5.py`. The six historical candidate
indicators form a 4/2 grouping: FD/PSV/SA/FN and SE/GTE. The second component is
limited to distribution/transition covariation, not complete direction randomness.
The X-ray classification experiments below are archived research history.

CPU setup (no GPU or deep-learning dependencies required):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[stats,raw,cfa,report]'
.venv/bin/python -m pytest tests/dual_factor -q
.venv/bin/python scripts/run_efa_v5.py --dry-run
```

The dry run audits the existing feature matrix without writing results. It works
from another working directory because configuration paths resolve from the project
root. Missing EDF-dependent tests explicitly skip; they are not raw-data validation.

For exploratory verification, first copy the full configuration and reduce iteration
counts in that copy if needed, then use `--diagnostic-only`. Such runs are always
`pending`, never accepted scientific results:

```bash
.venv/bin/python scripts/run_efa_v5.py --diagnostic-only
```

Full-size diagnostic calibration and latent-factor power are separate gates. A
matching 200-dataset/200-bootstrap diagnostic certificate can produce a method
correction report (B); positive latent acceptance (A) additionally requires validated
latent power. Observable covariance signal never waives PAF/C5. Small smoke runs
cannot issue valid certificates. Failures are retained, not repaired by relaxing thresholds.

```bash
.venv/bin/python scripts/validate_v5_simulation.py --output /absolute/path/to/new-validation.json
.venv/bin/python scripts/run_efa_v5.py --validation /absolute/path/to/new-validation.json
.venv/bin/python scripts/gen_biweekly_report.py --run-dir /absolute/path/to/run --output /absolute/path/to/new-report.docx
```

Outputs go to unique directories under `results/dual_factor_efa_v5/`; historical
matrices, reports and raw data are never overwritten. All later stages default to
`not_run`. The CFA diagnostic requires real EFA acceptance bound to the exact report
and matrix hashes, and ordinary ML diagnostics cannot promote evidence or launch
external validation. Prediction and external-data implementations remain gated.

Design and remaining work: [SPEC v5](docs/superpowers/specs/2026-11-08-dual-factor-efa-v5.md),
[Plan v5](docs/superpowers/plans/2026-11-08-dual-factor-efa-v5-plan.md),
[execution log](docs/dual_factor/v5_execution_log.md).

This repository supports a graduation thesis project on attention-state analysis for X-ray security inspection. The initial experiment studies whether gaze or error-distance sequences can distinguish alert and sleepy inspection states under easy and hard task difficulties.

## Data Policy

Raw data is not committed to this repository. The local folders `data/` and `Distan_error（原20）/` are treated as read-only inputs and are ignored by Git. Scripts and configuration may reference these folders, but they must not copy, rewrite, or version their contents.

## Environment

Create the reproducible Conda environment from the project root:

```bash
conda env create -f environment.yml
```

The environment name is `xray-attention` and it contains Python 3.11, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, PyYAML, and Pytest.

After the controller creates the environment, verify the package set with:

```bash
conda run -n xray-attention python -c "import numpy, pandas, scipy, sklearn, matplotlib, yaml; print('environment ready')"
```

Expected output:

```text
environment ready
```

## Threshold Experiment

The only parameter source for the threshold-selection experiment is `configs/threshold_selection.yaml`.

### Running the Experiment

```bash
# Run from the project root (the directory containing this README):
PYTHONPATH=src conda run -n xray-attention python experiments/threshold_selection/run.py --config configs/threshold_selection.yaml
```

### Outputs

Threshold-selection outputs are written under `results/threshold_selection/`:

- `config_used.yaml`: The exact configuration used for the experiment
- `task_features.csv`: 80 tasks, each with summary statistics and threshold features
- `candidate_summary.csv`: Aggregate performance across all candidate thresholds
- `nested_loso_folds.csv`: Per-fold results from the nested leave-one-subject-out evaluation
- `recommended_threshold.json`: Recommended attention thresholds:
  - `overall`: 900px
  - `easy`: 975px
  - `hard`: 625px
- `threshold_performance.png`: ROC-AUC curves across thresholds (optional)
- `selected_thresholds.png`: Distribution of selected thresholds across folds (optional)

The directory is tracked with `.gitkeep`; bulky intermediate arrays, videos, checkpoints, and raw artifacts are ignored. Lightweight summary files such as `.csv`, `.json`, `.md`, `.txt`, `.yaml`, and `.yml` remain trackable for thesis reporting and review.

## Labels

The label convention is fixed by `configs/threshold_selection.yaml`:

- `alert`: `1`
- `sleepy`: `0`
