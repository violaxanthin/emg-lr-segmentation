# EMG Late Response Segmentation

Notebook-first analysis pipeline for detecting, reviewing, and analyzing early,
middle, and late EMG responses around stimulation events.

The project started as exploratory notebooks and now keeps reusable analysis
logic in `src/emg_lr_segmentation/`, while notebooks remain the main interface
for running the workflow and reviewing signals interactively.

## What This Project Does

- Loads continuous EMG recordings from FIF or MATLAB `.mat` files.
- Detects stimulation events and response candidates.
- Supports manual ER/MR/LR calibration and interactive review in the MNE Qt
  browser.
- Saves reviewed annotations, response metrics, calibration summaries, and QC
  figures.
- Relates reviewed responses to movement-cycle phase in the final analysis
  notebook.

## Repository Layout

```text
.
├── notebooks/
│   ├── emg_lr_segmentation_1.ipynb
│   ├── emg_lr_segmentation_mat_2.ipynb
│   ├── manual_calibrated_er_mr_lr_3.ipynb
│   ├── movement_phase_analysis_4.ipynb
│   └── _archive/
├── src/
│   └── emg_lr_segmentation/
│       ├── manual_calibrated_er_mr_lr.py
│       └── movement_phase.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── annotations/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

`data/` and `outputs/` are local working directories. They are ignored by Git so
large recordings and generated analysis artifacts do not get committed.

## Workflow

Run notebooks from the repository root so relative paths resolve correctly.

1. `notebooks/emg_lr_segmentation_1.ipynb`
   Original FIF-based exploration.

2. `notebooks/emg_lr_segmentation_mat_2.ipynb`
   MATLAB `.mat` recording workflow.

3. `notebooks/manual_calibrated_er_mr_lr_3.ipynb`
   Current manual-calibrated ER/MR/LR detection and review pipeline. This
   notebook measures manual labels, calibrates candidate detection, opens the
   MNE browser for review, and writes reviewed metrics/annotations.

4. `notebooks/movement_phase_analysis_4.ipynb`
   Final movement-phase analysis. This notebook uses reviewed ER/MR/LR results
   and analyzes how responses relate to movement-cycle phase.

The first two notebooks are useful historical/context notebooks. The active
analysis path is usually notebooks 3 and 4.

## Python Modules

Reusable code lives under `src/emg_lr_segmentation/`.

- `manual_calibrated_er_mr_lr.py`
  Loading MATLAB recordings, extracting response metrics, building manual
  calibration, detecting ER/MR/LR candidates, normalizing reviewed annotations,
  and plotting reviewed epochs.

- `movement_phase.py`
  EMG envelope creation, movement phase extraction, phase binning, response
  summary tables, latency outlier marking, and movement-phase plots.

The notebooks add `src/` to `sys.path`, so editable package installation is not
required for normal notebook use.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The interactive review notebooks use MNE's Qt browser, so the environment needs
Qt support. The required packages include `PyQt6`, `mne-qt-browser`, and
`pyqtgraph`.

## Recording Paths And Data

Raw recordings are not included in the repository. Place local recordings in:

```text
data/raw/
```

or point the notebook directly to another local path.

Each notebook has a config cell near the top with the recording path. Replace
the placeholder path with your own recording file:

```python
# FIF workflow, notebook 1
fif_path = Path("my_path/my_dir/my_recording.fif")

# MATLAB workflows, notebooks 2-4
mat_path = Path("my_path/my_dir/my_recording.mat")
```

Supported recording formats are:

- `.fif` for `notebooks/emg_lr_segmentation_1.ipynb`
- `.mat` for `notebooks/emg_lr_segmentation_mat_2.ipynb`,
  `notebooks/manual_calibrated_er_mr_lr_3.ipynb`, and
  `notebooks/movement_phase_analysis_4.ipynb`

The active `.mat` workflow expects MATLAB files with the variables used by the
loader: `data`, `datastart`, `dataend`, `titles`, and `samplerate`. If your
recording uses different channel names, also update the channel settings in the
same config cell.

## Outputs

Generated files are written under:

```text
outputs/annotations/
outputs/figures/
outputs/metrics/
```

Typical outputs include:

- MNE annotation FIF files with reviewed ER/MR/LR labels.
- CSV metric tables before and after review.
- JSON calibration summaries.
- QC figures for response detection and movement-phase analysis.

These outputs are treated as reproducible local artifacts and are ignored by
Git.

## Interactive Review

Some notebooks intentionally open an MNE Qt browser for manual review. Close the
browser window to let the notebook continue.

For the manual-calibrated pipeline, you can skip the interactive review step in
automation by setting:

```bash
EMG_SKIP_INTERACTIVE_QC=1
```
