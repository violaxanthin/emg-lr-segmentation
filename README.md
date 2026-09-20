# EMG Late Response Segmentation

Analysis notebooks and helper code for detecting and reviewing early, middle,
and late EMG responses around stimulation events.

## Overview

This project works with continuous EMG recordings, identifies stimulation
artifacts, builds response-locked epochs, and supports manual calibration and
quality control of ER/MR/LR response candidates. The workflow is notebook-first,
with reusable helper functions kept in Python modules where possible.

## Data

Raw recordings are not included in this repository. Place local recordings under
`data/raw/` when running the notebooks. The `data/raw/`, `data/interim/`, and
`data/processed/` directories are ignored by Git.

Generated analysis outputs are also excluded from Git. Notebooks may write
annotations, metrics, JSON summaries, CSV tables, and figures under `outputs/`,
but those files are treated as reproducible local artifacts.

## Repository Structure

```text
notebooks/
  emg_lr_segmentation.ipynb
  emg_lr_segmentation_mat.ipynb
  manual_calibrated_er_mr_lr.ipynb
  movement_phase_analysis.ipynb
  manual_calibrated_er_mr_lr.py
  movement_phase_helpers.py
requirements.txt
```

The `_archive/` folder contains earlier notebook history. Local planning notes
under `.local/` are not part of the public repository.

## Setup

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The notebooks use MNE's Qt browser for interactive signal review, so the
requirements include `PyQt6`, `mne-qt-browser`, and `pyqtgraph`.

## Usage

Start with the notebook that matches the recording format you are analyzing:

- `notebooks/emg_lr_segmentation_mat.ipynb` for the MATLAB recording workflow.
- `notebooks/emg_lr_segmentation.ipynb` for the original FIF-based workflow.
- `notebooks/manual_calibrated_er_mr_lr.ipynb` for manual ER/MR/LR calibration
  and candidate review.
- `notebooks/movement_phase_analysis.ipynb` for movement-cycle phase analysis.

Run notebooks from the repository root so relative paths resolve consistently.

## Outputs

Generated files are written locally under `outputs/` and are intentionally not
tracked. Recreate them by rerunning the relevant notebooks after placing the
required source data in `data/raw/`.

## Notes

Some response-detection thresholds and review decisions are intentionally
calibrated interactively from representative epochs. Keep notebook outputs clear
before committing so repository diffs stay focused on source changes.
