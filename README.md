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
│       ├── er_mr_detection_review.py
│       ├── manual_calibrated_er_mr_lr.py
│       └── movement_phase.py
├── config/
│   └── er_morphology_thresholds.json
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── annotations/
│   ├── figures/
│   └── metrics/
├── tests/
│   └── test_core.py
├── requirements.txt
└── README.md
```

`data/` and `outputs/` are local working directories. They are ignored by Git so
large recordings, manual annotations, and generated analysis artifacts are not
committed. Detector settings that are required to start the workflow live in
the tracked `config/` directory instead.

## Workflow

Run notebooks from the repository root so relative paths resolve correctly.
For the MATLAB workflow, run notebooks 2, 3, and 4 in that order. Each notebook
creates the local files required by the next one.

1. `notebooks/emg_lr_segmentation_1.ipynb`
   Original FIF-based exploration.

2. `notebooks/emg_lr_segmentation_mat_2.ipynb`
   Loads the MATLAB recording, detects stimuli and response candidates, and
   saves the manually reviewed calibration annotations to
   `outputs/annotations/manual_er_mr_lr-annot.fif`.

3. `notebooks/manual_calibrated_er_mr_lr_3.ipynb`
   Reads the manual annotations from notebook 2, builds the ER/MR/LR
   calibration, opens the MNE browser for review, and writes reviewed metrics
   and annotations.

4. `notebooks/movement_phase_analysis_4.ipynb`
   Uses the calibration produced by notebook 3, performs the final ER/MR/LR
   review, and analyzes how the reviewed responses relate to movement-cycle
   phase.

Notebook 1 is the older FIF-based workflow and is independent of the active
MATLAB path.

## Python Modules

Reusable code lives under `src/emg_lr_segmentation/`.

- `manual_calibrated_er_mr_lr.py`
  Loading MATLAB recordings, extracting response metrics, building manual
  calibration, detecting ER/MR/LR candidates, normalizing reviewed annotations,
  and plotting reviewed epochs.

- `er_mr_detection_review.py`
  ER/MR detection, editable review, complete-pair validation, LR candidate
  search, and reviewed LR/MR amplitude ratios used by notebook 4.

- `movement_phase.py`
  EMG envelope creation, movement phase extraction, phase binning, response
  summary tables, latency outlier marking, and movement-phase plots.

The notebooks add `src/` to `sys.path`, so editable package installation is not
required for normal notebook use.

## Setup

Python 3.12 is recommended. Create and activate an environment, then install
the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The interactive review notebooks use MNE's Qt browser, so the environment needs
Qt support. The required packages include `PyQt6`, `mne-qt-browser`, and
`pyqtgraph`. Dependency versions are pinned in `requirements.txt` so the same
analysis environment can be recreated later.

Run the data-independent regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Recording Paths And Data

Raw recordings are not included in the repository. By default, the notebooks
look for:

```text
data/raw/recording.mat
data/raw/recording.fif
```

You can use a different location without editing a notebook by setting an
environment variable before starting Jupyter:

```bash
export EMG_MAT_PATH=/absolute/path/to/recording.mat
export EMG_FIF_PATH=/absolute/path/to/recording.fif
jupyter notebook
```

Supported recording formats are:

- `.fif` for `notebooks/emg_lr_segmentation_1.ipynb`
- `.mat` for `notebooks/emg_lr_segmentation_mat_2.ipynb`,
  `notebooks/manual_calibrated_er_mr_lr_3.ipynb`, and
  `notebooks/movement_phase_analysis_4.ipynb`

The active `.mat` workflow expects `data`, `datastart`, `dataend`, `titles`, and
`samplerate` variables. If the recording uses different channel names, update
the channel setting in the config cell near the top of each active notebook.

The initial ER morphology limits are stored in
`config/er_morphology_thresholds.json`. This file is an input to the analysis
and is version-controlled. Files under `outputs/` are generated locally and
must not be used as hidden configuration.

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
Git. Notebook cell outputs are also cleared before commits to avoid publishing
large embedded plots, local paths, or stale results. Running the notebooks will
recreate them.

## Interactive Review

Some notebooks intentionally open an MNE Qt browser for manual review. Close the
browser window to let the notebook continue.

For the manual-calibrated pipeline, you can skip the interactive review step in
automation by setting:

```bash
EMG_SKIP_INTERACTIVE_QC=1
```

This option applies to notebook 3. Notebooks 2 and 4 intentionally contain
manual review steps; results from those steps depend on the annotations chosen
by the reviewer.

## Reproducibility Changes

- Required ER morphology settings were moved from ignored `outputs/` into the
  tracked `config/` directory.
- Notebook 2 now saves its manual calibration file to the location consumed by
  notebook 3.
- Notebook 4 now starts from notebook 3's calibration instead of requiring
  private snapshots from a previous run or a fixed count of reviewed pairs.
- Recording paths can be supplied through `EMG_MAT_PATH` and `EMG_FIF_PATH`.
- Dependency versions are pinned, saved notebook outputs are cleared, and MNE
  display ranges have descriptive names. Display ranges affect only the browser
  view, not detector thresholds or signal values.
