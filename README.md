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

The planned command for the threshold experiment is:

```bash
python -m xray_attention.attention.threshold_selection --config configs/threshold_selection.yaml
```

Task 1 establishes the repository skeleton only; the executable experiment module will be added in a later task.

## Outputs

Threshold-selection outputs should be written under `results/threshold_selection/`. The directory is tracked with `.gitkeep`; bulky intermediate arrays, videos, checkpoints, and raw artifacts are ignored. Lightweight summary files such as `.csv`, `.json`, `.md`, `.txt`, `.yaml`, and `.yml` remain trackable for thesis reporting and review.

## Labels

The label convention is fixed by `configs/threshold_selection.yaml`:

- `alert`: `1`
- `sleepy`: `0`
