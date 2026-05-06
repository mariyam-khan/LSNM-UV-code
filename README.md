# LSNM-UV: Causal Discovery in Location-Scale Noise Models with Hidden Variables

This repository implements the LSNM-UV algorithm for identifying acyclic directed mixed graphs (ADMGs) from observational data under a location-scale noise model with hidden variables.

## Overview

LSNM-UV extends causal discovery beyond additive noise to handle **heteroscedastic** settings where causes modulate both the mean and variance of their effects. Our two-stage algorithm:

1. **LSNM-UV-Base** (Stage 1): Identifies parent-child relationships using location-scale residuals
2. **CheckVisible** (Stage 2): Re-examines invisible pairs to distinguish hidden confounders from incomplete parent sets

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing Experiments

Run all function configurations:
```bash
python run_experiments.py
```

Run a single configuration:
```bash
python run_experiments.py --config A --n-trials 25
```

## File Structure

| File | Description |
|------|-------------|
| `lsnm_uv_x.py` | LSNM-UV algorithm (Stage 1 + Stage 2) |
| `camuv_lsnm.py` | LSNM residual computation (location-scale GAM) |
| `camuv.py` | Original CAM-UV from Maeda & Shimizu (2021) |
| `lsnm_data_gen.py` | Data generation with 4 nonlinear function families |
| `run_experiments.py` | Experiment runner with multi-config support |
| `eval_metrics.py` | Evaluation metrics (precision, recall, F1) |
| `plot_summary_figure.py` | Generate the paper's summary figure |

## Citation

```bibtex
@inproceedings{lsnm-uv2026,
  title={Beyond Additivity: Causal Discovery in Location-Scale Noise Models with Hidden Variables},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```
