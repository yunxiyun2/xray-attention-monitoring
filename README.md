# X-ray Attention Monitoring

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
cd /Users/dyx/pythonproject/xray-attention-monitoring
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
