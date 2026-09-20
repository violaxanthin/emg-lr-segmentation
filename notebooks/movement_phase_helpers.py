"""Helpers for estimating movement-cycle phase from continuous EMG.

This module is intentionally separate from the ER/MR/LR detectors.  The phase
pipeline is a preprocessing branch for continuous movement modulation, not a
replacement for stimulus-evoked response detection.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def _as_1d_signal(signal, name):
    """Return a finite 1D float array or raise a clear error."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"{name} must be a 1D continuous signal, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} contains NaNs or infinite values")
    return signal


def validate_filter_parameters(
    sfreq,
    emg_band=(30.0, 500.0),
    envelope_lowpass=5.0,
    movement_band=(0.5, 2.5),
):
    """Validate filter cutoffs without silently changing them.

    Returns a small dictionary with ``sfreq`` and ``nyquist`` so notebook cells
    can print exactly what was checked.
    """
    sfreq = float(sfreq)
    nyquist = sfreq / 2.0
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sampling frequency: sfreq={sfreq!r}")

    emg_low = emg_high = None
    if emg_band is not None:
        emg_low, emg_high = map(float, emg_band)

    move_low = move_high = None
    if movement_band is not None:
        move_low, move_high = map(float, movement_band)

    if envelope_lowpass is not None:
        envelope_lowpass = float(envelope_lowpass)

    checks = []
    if emg_band is not None:
        checks.extend([
            ("emg_band lower cutoff", emg_low),
            ("emg_band upper cutoff", emg_high),
        ])
    if envelope_lowpass is not None:
        checks.append(("envelope_lowpass", envelope_lowpass))
    if movement_band is not None:
        checks.extend([
            ("movement_band lower cutoff", move_low),
            ("movement_band upper cutoff", move_high),
        ])
    for label, value in checks:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be positive and finite, got {value!r}")

    if emg_band is not None and emg_low >= emg_high:
        raise ValueError(f"emg_band must be low < high, got {emg_band!r}")
    if movement_band is not None and move_low >= move_high:
        raise ValueError(f"movement_band must be low < high, got {movement_band!r}")

    invalid = []
    if emg_band is not None:
        invalid.append(("emg_band upper cutoff", emg_high))
    if envelope_lowpass is not None:
        invalid.append(("envelope_lowpass", envelope_lowpass))
    if movement_band is not None:
        invalid.append(("movement_band upper cutoff", move_high))
    for label, value in invalid:
        if value >= nyquist:
            raise ValueError(
                f"{label}={value:g} Hz is invalid for sfreq={sfreq:g} Hz. "
                f"Nyquist is {nyquist:g} Hz, and filter cutoffs must be strictly below Nyquist."
            )

    if movement_band is not None and envelope_lowpass is not None and move_high >= envelope_lowpass:
        warnings.warn(
            "movement_band upper cutoff is not below envelope_lowpass. "
            "This is allowed, but it weakens the intended separation between "
            "general EMG smoothing and movement-cycle isolation.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "sfreq": sfreq,
        "nyquist": nyquist,
        "emg_band": None if emg_band is None else (emg_low, emg_high),
        "envelope_lowpass": envelope_lowpass,
        "movement_band": None if movement_band is None else (move_low, move_high),
    }


def _sos_filter(signal, sfreq, cutoff, btype, filter_order=4):
    signal = _as_1d_signal(signal, "signal")
    sos = butter(filter_order, cutoff, btype=btype, fs=sfreq, output="sos")
    try:
        return sosfiltfilt(sos, signal, axis=-1)
    except ValueError as exc:
        raise ValueError(
            "Filtering failed. The recording may be too short for zero-phase "
            "sosfiltfilt padding, or the filter parameters may be incompatible "
            f"with sfreq={sfreq:g} Hz."
        ) from exc


def make_emg_envelope(raw_emg, sfreq, emg_band=(30.0, 500.0), envelope_lowpass=5.0, filter_order=4):
    """Create a slow EMG envelope from continuous raw EMG.

    Inputs
    ------
    raw_emg : array, shape (n_times,)
        Continuous EMG from one channel.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    filtered_emg, rectified_emg, envelope : arrays, each shape (n_times,)
    """
    validate_filter_parameters(
        sfreq,
        emg_band=emg_band,
        envelope_lowpass=envelope_lowpass,
        movement_band=None,
    )
    raw_emg = _as_1d_signal(raw_emg, "raw_emg")
    filtered_emg = _sos_filter(raw_emg, sfreq, emg_band, "bandpass", filter_order)
    rectified_emg = np.abs(filtered_emg)
    envelope = _sos_filter(rectified_emg, sfreq, envelope_lowpass, "lowpass", filter_order)
    return filtered_emg, rectified_emg, envelope


def extract_movement_signal(envelope, sfreq, movement_band=(0.5, 2.5), filter_order=4):
    """Isolate the slow oscillatory envelope component used as movement signal."""
    validate_filter_parameters(
        sfreq,
        emg_band=None,
        envelope_lowpass=None,
        movement_band=movement_band,
    )
    envelope = _as_1d_signal(envelope, "envelope")
    return _sos_filter(envelope, sfreq, movement_band, "bandpass", filter_order)


def compute_movement_phase(movement_signal):
    """Compute Hilbert amplitude and phase from a narrow-band movement signal."""
    movement_signal = _as_1d_signal(movement_signal, "movement_signal")
    analytic_signal = hilbert(movement_signal)
    hilbert_amplitude = np.abs(analytic_signal)
    movement_phase = np.angle(analytic_signal)
    return analytic_signal, hilbert_amplitude, movement_phase


def detect_cycle_boundaries(movement_phase, wrap_threshold=-np.pi):
    """Find initial cycle boundaries from large negative phase wraps."""
    movement_phase = _as_1d_signal(movement_phase, "movement_phase")
    phase_step = np.diff(movement_phase)
    return np.where(phase_step < wrap_threshold)[0] + 1


def assign_phase_to_stimuli(movement_phase, stim_samples, sfreq, first_samp=0):
    """Assign continuous movement phase in radians to stimulus sample indices."""
    movement_phase = _as_1d_signal(movement_phase, "movement_phase")
    stim_samples = np.asarray(stim_samples, dtype=int)
    first_samp = int(first_samp)
    if stim_samples.ndim != 1:
        raise ValueError(f"stim_samples must be 1D, got shape {stim_samples.shape}")
    stim_indices = stim_samples - first_samp
    if stim_indices.size and (stim_indices.min() < 0 or stim_indices.max() >= movement_phase.size):
        raise ValueError(
            "stim_samples are outside movement_phase bounds: "
            f"valid raw sample range is {first_samp}..{first_samp + movement_phase.size - 1}, got "
            f"{stim_samples.min()}..{stim_samples.max()}"
        )
    return {
        "stimulus_sample": stim_samples,
        "stimulus_time": stim_indices / float(sfreq),
        "movement_phase": movement_phase[stim_indices],
    }
