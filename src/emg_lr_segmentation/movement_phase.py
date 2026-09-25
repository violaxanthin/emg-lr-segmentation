"""Helpers for estimating movement-cycle phase from continuous EMG.

This module is intentionally separate from the ER/MR/LR detectors.  The phase
pipeline is a preprocessing branch for continuous movement modulation, not a
replacement for stimulus-evoked response detection.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

DEFAULT_PHASE_CMAP = "tab10"
DEFAULT_COMPONENT_COLORS = {"er": "tab:green", "mr": "tab:orange", "lr": "tab:blue"}
DEFAULT_RESPONSE_PREFIX = {"er": "early", "mr": "middle", "lr": "late"}
DEFAULT_RESPONSE_LABEL_ORDER = ["early", "middle", "late"]
DEFAULT_LR_VS_M_METRICS = [
    ("late_p2p_amplitude", "LR p2p amplitude"),
    ("late_rms", "LR RMS"),
    ("late_auc_abs", "LR absolute AUC"),
]


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


def make_phase_bins(n_phase_bins):
    """Return equal phase-bin edges and centers over ``-pi..+pi``."""
    edges = np.linspace(-np.pi, np.pi, int(n_phase_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def assign_phase_bins(stim_phases, phase_bin_edges):
    """Assign one human-readable phase bin to each stimulus."""
    result = stim_phases.copy()
    n_bins = len(phase_bin_edges) - 1
    phase = pd.to_numeric(result["movement_phase"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(phase)

    adjusted = phase.copy()
    adjusted[valid & np.isclose(adjusted, np.pi)] = np.nextafter(np.pi, -np.inf)
    bins = np.full(len(result), np.nan)
    bins[valid] = np.digitize(adjusted[valid], phase_bin_edges, right=False)
    valid_bins = valid & (bins >= 1) & (bins <= n_bins)

    result["phase_bin"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[valid_bins, "phase_bin"] = bins[valid_bins].astype(int)
    centers = pd.Series({
        bin_id: center
        for bin_id, center in zip(range(1, n_bins + 1), (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2)
    })
    result["phase_bin_center"] = result["phase_bin"].map(centers).astype(float)
    result["phase_sin"] = np.sin(result["movement_phase"])
    result["phase_cos"] = np.cos(result["movement_phase"])
    return result


def validate_phase_alignment(movement_phase, stim_samples, raw, sfreq):
    """Check that stimulus samples can index the continuous movement phase."""
    stim_samples = np.asarray(stim_samples)
    stim_indices = stim_samples - raw.first_samp
    integer_samples = np.issubdtype(stim_samples.dtype, np.integer)
    within_bounds = (stim_indices >= 0) & (stim_indices < len(movement_phase))
    finite_phase = (
        np.isfinite(movement_phase[stim_indices[within_bounds]])
        if within_bounds.any()
        else np.array([], dtype=bool)
    )

    summary = pd.DataFrame([{
        "movement_phase_samples": len(movement_phase),
        "raw_samples": raw.n_times,
        "one_phase_per_raw_sample": len(movement_phase) == raw.n_times,
        "stimuli_total": len(stim_samples),
        "stimulus_samples_integer": bool(integer_samples),
        "stimuli_within_phase_bounds": int(within_bounds.sum()),
        "stimuli_out_of_bounds": int((~within_bounds).sum()),
        "stimuli_with_finite_phase": int(finite_phase.sum()),
        "raw_first_samp": int(raw.first_samp),
        "sfreq": float(sfreq),
    }])

    if not bool(summary.loc[0, "one_phase_per_raw_sample"]):
        raise ValueError("movement_phase does not have one value per raw sample")
    if not integer_samples:
        raise TypeError("stim_samples must be integer sample indices")
    if not within_bounds.all():
        raise ValueError("Some stimulus samples are outside movement_phase bounds")
    if finite_phase.sum() != len(stim_samples):
        raise ValueError("Some stimulus samples have missing or non-finite movement phase")
    return summary


def _first_response_per_stimulus(reviewed_metrics, label):
    """Return the first reviewed response of one label per stimulus."""
    subset = reviewed_metrics.loc[reviewed_metrics["label"].eq(label)].copy()
    if subset.empty:
        return subset
    sort_cols = ["stimulus_index", "component_rank", "peak_latency_ms"]
    sort_cols = [col for col in sort_cols if col in subset.columns]
    return subset.sort_values(sort_cols).groupby("stimulus_index", as_index=False).first()


def _strongest_lr_per_stimulus(reviewed_metrics):
    """Return the strongest reviewed LR row per stimulus."""
    subset = reviewed_metrics.loc[reviewed_metrics["label"].eq("lr")].copy()
    if subset.empty:
        return subset
    sort_cols = ["stimulus_index", "prominence", "absolute_peak_amplitude", "peak_latency_ms"]
    ascending = [True, False, False, True]
    return subset.sort_values(sort_cols, ascending=ascending).groupby("stimulus_index", as_index=False).first()


def build_response_by_stim(
    stim_phases,
    reviewed_metrics,
    metric_columns,
    response_prefix=DEFAULT_RESPONSE_PREFIX,
):
    """Create one row per stimulation with phase and reviewed ER/MR/LR metrics."""
    base = stim_phases.rename(columns={"stimulus_time": "stimulus_time_s"}).copy()
    base.insert(0, "stimulus_id", base["stimulus_index"].astype(int))
    response_by_stim = base.copy()

    for label, prefix in response_prefix.items():
        component = (
            _strongest_lr_per_stimulus(reviewed_metrics)
            if label == "lr"
            else _first_response_per_stimulus(reviewed_metrics, label)
        )
        keep = ["stimulus_index", *[col for col in metric_columns if col in component.columns]]
        keep += [col for col in ["start_s", "end_s", "annotation_onset_s", "component_rank"] if col in component.columns]
        component = component.loc[:, list(dict.fromkeys(keep))].copy() if len(component) else pd.DataFrame(columns=keep)
        component = component.rename(columns={col: f"{prefix}_{col}" for col in component.columns if col != "stimulus_index"})
        response_by_stim = response_by_stim.merge(component, on="stimulus_index", how="left", validate="one_to_one")
        present_col = f"{prefix}_peak_latency_ms"
        response_by_stim[f"{prefix}_present"] = response_by_stim[present_col].notna() if present_col in response_by_stim else False

    counts = reviewed_metrics.pivot_table(index="stimulus_index", columns="label", values="peak_latency_ms", aggfunc="size", fill_value=0)
    for label, prefix in response_prefix.items():
        label_counts = counts[label] if label in counts else pd.Series(dtype=int)
        response_by_stim[f"{prefix}_count"] = response_by_stim["stimulus_index"].map(label_counts).fillna(0).astype(int)

    response_by_stim["phase_at_er"] = response_by_stim["movement_phase"].where(response_by_stim["early_present"])
    response_by_stim["phase_at_mr"] = response_by_stim["movement_phase"].where(response_by_stim["middle_present"])
    response_by_stim["phase_at_lr"] = response_by_stim["movement_phase"].where(response_by_stim["late_present"])
    return response_by_stim


def summarize_by_phase_bin(
    response_by_stim,
    metric_columns,
    n_phase_bins,
    response_label_order=DEFAULT_RESPONSE_LABEL_ORDER,
):
    """Summarize per-stimulus response metrics by phase bin."""
    bin_labels = np.arange(1, int(n_phase_bins) + 1)
    grouped = response_by_stim.groupby("phase_bin", observed=False)
    summary = grouped.size().rename("total_stim_count").reindex(bin_labels, fill_value=0).to_frame()

    continuous_cols = []
    for prefix in response_label_order:
        for metric in metric_columns:
            col = f"{prefix}_{metric}"
            if col in response_by_stim.columns:
                continuous_cols.append(col)

    if continuous_cols:
        stats = grouped[continuous_cols].agg(["count", "mean", "median", "std"]).reindex(bin_labels)
        stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
        summary = summary.join(stats)

    late_count = grouped["late_present"].sum().reindex(bin_labels, fill_value=0).astype(int)
    summary["late_response_count"] = late_count
    summary["late_response_fraction"] = summary["late_response_count"] / summary["total_stim_count"].replace(0, np.nan)
    return summary.reset_index().rename(columns={"phase_bin": "phase_bin"})


def add_lr_latency_outliers(response_by_stim, value_col="late_peak_latency_ms", mad_k=3.0):
    """Mark robust outliers in LR latency without excluding data."""
    result = response_by_stim.copy()
    values = pd.to_numeric(result[value_col], errors="coerce")
    center = float(values.median())
    mad = float((values - center).abs().median())
    scaled_mad = 1.4826 * mad if np.isfinite(mad) else np.nan
    if not np.isfinite(scaled_mad) or scaled_mad == 0:
        result["late_latency_robust_z"] = np.nan
        result["late_latency_outlier"] = False
        return result, center, scaled_mad

    robust_z = (values - center) / scaled_mad
    result["late_latency_robust_z"] = robust_z
    result["late_latency_outlier"] = robust_z.abs() > mad_k
    return result, center, scaled_mad


def _phase_scatter(
    ax,
    data,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    n_phase_bins,
    phase_cmap=DEFAULT_PHASE_CMAP,
):
    """Draw a phase-colored scatter plot for one x/y metric pair."""
    plot_df = data.dropna(subset=[x_col, y_col, "phase_bin"]).copy()
    if plot_df.empty:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return None
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df["phase_bin"].astype(float),
        cmap=phase_cmap,
        vmin=1,
        vmax=n_phase_bins,
        s=36,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    return scatter


def plot_raw_envelope_phase_segment(
    time_s,
    raw_emg,
    envelope,
    movement_signal,
    movement_phase,
    response_by_stim,
    cycle_boundaries,
    sfreq,
    start_s,
    stop_s,
    n_phase_bins=8,
    phase_cmap=DEFAULT_PHASE_CMAP,
):
    """Plot raw EMG, envelope/movement signal, phase, and stimulus markers."""
    mask = (time_s >= start_s) & (time_s <= stop_s)
    if mask.sum() < 2:
        raise ValueError(f"QC window {start_s}..{stop_s} s does not overlap the recording")

    fig, axes = plt.subplots(4, 1, figsize=(15, 7), sharex=True, height_ratios=[1, 1, 1, 1])
    axes[0].plot(time_s[mask], raw_emg[mask], color="0.60", linewidth=0.7)
    axes[0].set_ylabel("raw GM R", rotation=0, ha="right", va="center")
    axes[1].plot(time_s[mask], envelope[mask], color="tab:blue", linewidth=1.1)
    axes[1].set_ylabel("envelope", rotation=0, ha="right", va="center")
    axes[2].plot(time_s[mask], movement_signal[mask], color="tab:cyan", linewidth=1.0)
    axes[2].set_ylabel("movement", rotation=0, ha="right", va="center")
    axes[3].plot(time_s[mask], movement_phase[mask], color="tab:purple", linewidth=1.0)
    axes[3].set_ylabel("phase", rotation=0, ha="right", va="center")
    axes[3].set_yticks([-np.pi, 0, np.pi])
    axes[3].set_yticklabels(["-pi", "0", "+pi"])

    boundary_times = cycle_boundaries / sfreq
    for boundary_s in boundary_times[(boundary_times >= start_s) & (boundary_times <= stop_s)]:
        axes[2].axvline(boundary_s, color="0.2", alpha=0.25, linewidth=0.8)
        axes[3].axvline(boundary_s, color="0.2", alpha=0.25, linewidth=0.8)

    qc_stims = response_by_stim.loc[response_by_stim["stimulus_time_s"].between(start_s, stop_s)]
    cmap = plt.get_cmap(phase_cmap, n_phase_bins)
    for row in qc_stims.itertuples():
        color = cmap((int(row.phase_bin) - 1) % n_phase_bins) if pd.notna(row.phase_bin) else "tab:red"
        for ax in axes:
            ax.axvline(row.stimulus_time_s, color=color, alpha=0.55, linewidth=0.9)
        axes[3].text(row.stimulus_time_s, np.pi, str(row.phase_bin), ha="center", va="bottom", fontsize=7, color=color)

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time, s")
    fig.suptitle(f"Raw EMG, envelope, movement signal, and phase: {start_s:g}-{stop_s:g} s")
    plt.tight_layout()
    return fig, axes


def plot_lr_vs_m_response(
    response_by_stim,
    lr_vs_m_metrics=DEFAULT_LR_VS_M_METRICS,
    n_phase_bins=8,
    phase_cmap=DEFAULT_PHASE_CMAP,
):
    """Plot LR amplitude/energy metrics against M-response amplitude."""
    fig, axes = plt.subplots(1, len(lr_vs_m_metrics), figsize=(5.2 * len(lr_vs_m_metrics), 4), squeeze=False)
    plotted_phase_bins = set()
    for ax, (y_col, y_label) in zip(axes[0], lr_vs_m_metrics):
        _phase_scatter(
            ax,
            response_by_stim,
            "middle_p2p_amplitude",
            y_col,
            y_label,
            "M-response p2p amplitude",
            y_label,
            n_phase_bins=n_phase_bins,
            phase_cmap=phase_cmap,
        )
        plotted_phase_bins.update(response_by_stim.dropna(subset=["middle_p2p_amplitude", y_col, "phase_bin"])["phase_bin"].astype(int).tolist())
    if plotted_phase_bins:
        cmap = plt.get_cmap(phase_cmap, n_phase_bins)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap((phase_bin - 1) / max(n_phase_bins - 1, 1)), markeredgecolor="white", markersize=7, label=f"bin {phase_bin}")
            for phase_bin in sorted(plotted_phase_bins)
        ]
        axes[0, -1].legend(handles=handles, title="Phase bin", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.suptitle("Late response vs M-response, colored by movement phase")
    plt.tight_layout()
    return fig, axes


def plot_lr_latency_outliers(
    response_by_stim,
    latency_center_ms,
    latency_scale_ms,
    mad_k=3.0,
    n_phase_bins=8,
    phase_cmap=DEFAULT_PHASE_CMAP,
):
    """Plot LR latency by phase bin and highlight robust outliers."""
    phase_bin_labels = np.arange(1, int(n_phase_bins) + 1)
    plot_df = response_by_stim.dropna(subset=["late_peak_latency_ms", "phase_bin"]).copy()
    plot_df["phase_bin"] = plot_df["phase_bin"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bin_values = [
        plot_df.loc[plot_df["phase_bin"].eq(phase_bin), "late_peak_latency_ms"].to_numpy()
        for phase_bin in phase_bin_labels
    ]
    non_empty_positions = [int(phase_bin) for phase_bin, values in zip(phase_bin_labels, bin_values) if len(values)]
    non_empty_values = [values for values in bin_values if len(values)]
    if non_empty_values:
        ax.boxplot(
            non_empty_values,
            positions=non_empty_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "0.88", "edgecolor": "0.35"},
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "0.35"},
            capprops={"color": "0.35"},
        )
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap(phase_cmap, n_phase_bins)
    for phase_bin in phase_bin_labels:
        bin_df = plot_df.loc[plot_df["phase_bin"].eq(int(phase_bin))]
        if bin_df.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(bin_df))
        ax.scatter(
            np.full(len(bin_df), int(phase_bin)) + jitter,
            bin_df["late_peak_latency_ms"],
            color=cmap((int(phase_bin) - 1) / max(n_phase_bins - 1, 1)),
            edgecolor="white",
            linewidth=0.4,
            s=40,
            alpha=0.75,
        )
    outliers = plot_df.loc[plot_df["late_latency_outlier"]]
    if len(outliers):
        ax.scatter(
            outliers["phase_bin"].astype(int),
            outliers["late_peak_latency_ms"],
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            s=90,
            label="robust latency outlier",
        )
        for row in outliers.itertuples():
            ax.annotate(str(row.stimulus_index), (int(row.phase_bin), row.late_peak_latency_ms), xytext=(4, 4), textcoords="offset points", fontsize=8)
    if np.isfinite(latency_scale_ms) and latency_scale_ms > 0:
        ax.axhline(latency_center_ms, color="black", linewidth=1.0, label="latency median")
        ax.axhspan(latency_center_ms - mad_k * latency_scale_ms, latency_center_ms + mad_k * latency_scale_ms, color="0.7", alpha=0.18, label="median +/- 3 scaled MAD")
    ax.set_title("LR latency by movement phase bin")
    ax.set_xlabel("Phase bin")
    ax.set_ylabel("LR peak latency, ms")
    ax.set_xticks(phase_bin_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax, plot_df


def plot_phase_bin_summary(response_by_stim, response_by_phase_bin, n_phase_bins=8):
    """Plot M-response, LR strength, LR frequency, and stimulus count by phase bin."""
    phase_bin_labels = np.arange(1, int(n_phase_bins) + 1)
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, height_ratios=[2, 2, 1.5, 1])
    plot_specs = [
        ("middle_p2p_amplitude", "M-response p2p amplitude", axes[0]),
        ("late_p2p_amplitude", "LR p2p amplitude", axes[1]),
    ]
    rng = np.random.default_rng(7)
    for col, ylabel, ax in plot_specs:
        count_col = f"{col}_count"
        count_by_bin = (
            response_by_phase_bin.set_index("phase_bin")[count_col].reindex(phase_bin_labels).fillna(0).astype(int)
            if count_col in response_by_phase_bin
            else pd.Series(0, index=phase_bin_labels)
        )
        for phase_bin, count in count_by_bin.items():
            if count == 0:
                ax.axvspan(float(phase_bin) - 0.45, float(phase_bin) + 0.45, color="0.85", alpha=0.35, zorder=0)
        values_df = response_by_stim.dropna(subset=["phase_bin", col]).copy()
        if len(values_df):
            jitter = rng.uniform(-0.08, 0.08, len(values_df))
            ax.scatter(values_df["phase_bin"].astype(float) + jitter, values_df[col], s=24, alpha=0.55, color="0.35")
            med = values_df.groupby("phase_bin")[col].median().reindex(phase_bin_labels)
            ax.plot(phase_bin_labels, med, color="tab:red", marker="o", linewidth=1.5, label="median")
            y_min, y_max = ax.get_ylim()
            label_y = y_max - 0.06 * (y_max - y_min)
            for phase_bin, count in count_by_bin.items():
                if count > 0:
                    ax.text(float(phase_bin), label_y, f"n={count}", ha="center", va="top", fontsize=8, color="0.25")
            ax.legend(loc="best")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[2].plot(response_by_phase_bin["phase_bin"], response_by_phase_bin["late_response_fraction"], color="tab:blue", marker="o", linewidth=1.5)
    axes[2].set_ylabel("LR frequency")
    axes[2].set_ylim(0, max(0.05, np.nanmax(response_by_phase_bin["late_response_fraction"]) * 1.2))
    axes[2].grid(alpha=0.25)

    axes[3].bar(response_by_phase_bin["phase_bin"], response_by_phase_bin["total_stim_count"], color="0.55")
    axes[3].set_ylabel("stimuli")
    axes[3].set_xlabel("Movement phase bin")
    axes[3].set_xticks(phase_bin_labels)
    axes[3].grid(axis="y", alpha=0.25)

    fig.suptitle("M-response, LR, and sample count by movement phase")
    plt.tight_layout()
    return fig, axes


def plot_phase_at_response_counts(
    response_by_stim,
    n_phase_bins=8,
    component_colors=DEFAULT_COMPONENT_COLORS,
):
    """Plot phase-bin counts for stimuli where ER, MR, or LR is present."""
    phase_bin_labels = np.arange(1, int(n_phase_bins) + 1)
    specs = [
        ("early_present", "phase_at_er", "ER"),
        ("middle_present", "phase_at_mr", "MR"),
        ("late_present", "phase_at_lr", "LR"),
    ]
    count_tables = []
    for present_col, phase_col, label in specs:
        counts = (
            response_by_stim.loc[response_by_stim[present_col], "phase_bin"]
            .value_counts()
            .reindex(phase_bin_labels, fill_value=0)
            .sort_index()
        )
        count_tables.append((label, counts))
    ymax = max([counts.max() for _, counts in count_tables] + [1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (label, counts) in zip(axes, count_tables):
        ax.bar(counts.index, counts.values, color=component_colors[label.lower()])
        ax.set_title(label)
        ax.set_xlabel("Phase bin")
        ax.set_xticks(phase_bin_labels)
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Stimuli with response")
    fig.suptitle("phase_at_er / phase_at_mr / phase_at_lr by movement phase bin")
    plt.tight_layout()
    return fig, axes


def plot_10_epochs_by_phase(
    response_by_stim,
    phase_response_metrics,
    raw_emg,
    time_s,
    sfreq,
    n_plots=10,
    component_colors=DEFAULT_COMPONENT_COLORS,
):
    """Plot up to ten LR-containing epochs with phase and metric annotations."""
    rank_df = response_by_stim.loc[response_by_stim["late_present"]].copy()
    if rank_df.empty:
        return None, rank_df
    rank_df["latency_rank"] = rank_df["late_latency_outlier"].astype(int) if "late_latency_outlier" in rank_df else 0
    sort_cols = ["latency_rank", "late_prominence", "late_absolute_peak_amplitude"]
    sort_cols = [col for col in sort_cols if col in rank_df.columns]
    rank_df = rank_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n_plots)

    fig, axes = plt.subplots(len(rank_df), 1, figsize=(10, max(2.0 * len(rank_df), 4)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for plot_i, (_, stim_row) in enumerate(rank_df.iterrows()):
        ax = axes[plot_i]
        stim_s = float(stim_row["stimulus_time_s"])
        stim_idx = int(stim_row["stimulus_index"])
        rows = phase_response_metrics.loc[phase_response_metrics["stimulus_index"].eq(stim_idx)].sort_values(["start_s", "label"])
        right_ms = max(70.0, float(rows["end_latency_ms"].max()) + 8.0 if len(rows) else 70.0)
        left_s = max(0.0, stim_s - 0.005)
        right_s = min(time_s[-1], stim_s + right_ms / 1000.0)
        mask = (time_s >= left_s) & (time_s <= right_s)
        x_ms = (time_s[mask] - stim_s) * 1000
        ax.plot(x_ms, raw_emg[mask], color="black", linewidth=0.9)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        for response in rows.itertuples():
            color = component_colors.get(str(response.label), "0.5")
            label = str(response.label).upper() if plot_i == 0 else None
            ax.axvspan(float(response.start_latency_ms), float(response.end_latency_ms), color=color, alpha=0.18, label=label)
            ax.axvline(float(response.peak_latency_ms), color=color, linewidth=0.8, alpha=0.9)
        outlier_mark = " | latency outlier" if bool(stim_row.get("late_latency_outlier", False)) else ""
        ax.set_title(
            f"stim {stim_idx} | bin {stim_row['phase_bin']} | phase {stim_row['movement_phase']:.2f} | "
            f"M p2p {stim_row.get('middle_p2p_amplitude', np.nan):.3g} | LR p2p {stim_row.get('late_p2p_amplitude', np.nan):.3g} | "
            f"LR lat {stim_row.get('late_peak_latency_ms', np.nan):.1f} ms{outlier_mark}",
            loc="left",
            fontsize=8,
        )
        ax.grid(alpha=0.25, linestyle=":")
    axes[-1].set_xlabel("Time after stimulus, ms")
    axes[0].legend(loc="upper right", ncols=3)
    fig.text(0.02, 0.5, "Amplitude, native", rotation="vertical", va="center")
    fig.suptitle("Ten LR-containing epochs: waveform check for phase and latency")
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])
    return fig, rank_df
