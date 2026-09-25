"""Utilities for the manual-calibrated ER/MR/LR notebook.

The functions in this module deliberately keep file I/O and the interactive MNE
browser in the notebook.  This makes metric extraction and candidate detection
repeatable and testable without opening a GUI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, peak_prominences, peak_widths


COMPONENTS = ("er", "mr", "lr")
AUTO_LABELS = {"er": "er_auto", "mr": "mr_auto", "lr": "lr_auto"}


def load_mat_raw(mat_path: Path) -> mne.io.BaseRaw:
    """Load and concatenate all blocks in the project MATLAB recording."""
    mat = loadmat(
        mat_path,
        variable_names=["data", "datastart", "dataend", "titles", "samplerate"],
        simplify_cells=True,
    )
    data = mat["data"].ravel()
    starts = mat["datastart"].astype(int) - 1
    ends = mat["dataend"].astype(int)
    ch_names = [str(name).strip() for name in mat["titles"]]
    sfreq = float(mat["samplerate"][0, 0])
    blocks = []
    for block_idx in range(starts.shape[1]):
        block_data = np.vstack(
            [
                data[starts[ch_idx, block_idx] : ends[ch_idx, block_idx]]
                for ch_idx in range(len(ch_names))
            ]
        )
        info = mne.create_info(ch_names, sfreq, ch_types="emg")
        blocks.append(mne.io.RawArray(block_data, info, verbose=False))
    return mne.concatenate_raws(blocks, preload=True, verbose=False)


def robust_center_scale(values) -> tuple[float, float]:
    """Return median and scaled MAD, with an IQR/sample-resolution fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        q25, q75 = np.percentile(values, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        nonzero = np.abs(values - center)
        scale = float(np.min(nonzero[nonzero > 0])) if np.any(nonzero > 0) else 0.25
    return center, scale


def _peak_shape(oriented, local_peak):
    """Return prominence and half-prominence bounds without edge warnings."""
    oriented = np.asarray(oriented, dtype=float)
    if len(oriented) < 3 or local_peak <= 0 or local_peak >= len(oriented) - 1:
        prominence = max(0.0, float(oriented[local_peak] - np.min(oriented)))
        half_width = max(1.0, len(oriented) / 2)
        left = max(0.0, local_peak - half_width / 2)
        right = min(float(len(oriented) - 1), local_peak + half_width / 2)
        return prominence, half_width, left, right
    prominence = float(peak_prominences(oriented, [local_peak])[0][0])
    widths, _, left_ips, right_ips = peak_widths(oriented, [local_peak], rel_height=0.5)
    return prominence, float(widths[0]), float(left_ips[0]), float(right_ips[0])


def _nearest_stimulus(onset_s: float, stimulus_times_s: np.ndarray):
    idx = int(np.searchsorted(stimulus_times_s, onset_s, side="right") - 1)
    if idx < 0:
        return None
    return idx, float(stimulus_times_s[idx])


def _baseline(signal, sfreq, stim_s, baseline_window_ms):
    start_s = stim_s + baseline_window_ms[0] / 1000
    end_s = stim_s + baseline_window_ms[1] / 1000
    i0 = max(0, int(round(start_s * sfreq)))
    i1 = min(len(signal), max(i0 + 1, int(round(end_s * sfreq))))
    segment = signal[i0:i1]
    median = float(np.median(segment))
    noise = float(1.4826 * np.median(np.abs(segment - median)))
    return median, max(noise, np.finfo(float).eps)


def _dominant_polarities(annotation_df, signal, sfreq, stimulus_times_s, baseline_window_ms):
    signs = {label: [] for label in COMPONENTS}
    radius = max(1, int(round(0.75 / 1000 * sfreq)))
    for row in annotation_df.itertuples():
        match = _nearest_stimulus(float(row.onset_s), stimulus_times_s)
        if match is None:
            continue
        _, stim_s = match
        baseline, _ = _baseline(signal, sfreq, stim_s, baseline_window_ms)
        i = int(round(float(row.onset_s) * sfreq))
        if row.duration_s > 0:
            i1 = max(i + 1, int(round((row.onset_s + row.duration_s) * sfreq)))
            local = signal[max(0, i) : min(len(signal), i1)] - baseline
        else:
            local = signal[max(0, i - radius) : min(len(signal), i + radius + 1)] - baseline
        if len(local):
            extreme = float(local[np.argmax(np.abs(local))])
            signs[row.label].append(np.sign(extreme))
    return {
        label: int(1 if np.nansum(signs[label]) >= 0 else -1)
        for label in COMPONENTS
    }


def _typical_durations(annotation_df):
    durations = {}
    for label in COMPONENTS:
        positive = (
            annotation_df.loc[
                (annotation_df["label"] == label) & (annotation_df["duration_s"] > 0),
                "duration_s",
            ].to_numpy()
            * 1000
        )
        if len(positive):
            center, scale = robust_center_scale(positive)
            clean = positive[np.abs(positive - center) <= 3 * scale]
            durations[label] = float(np.median(clean if len(clean) else positive))
        else:
            durations[label] = 2.0
    return durations


def _measure_wave(
    signal,
    sfreq,
    stim_s,
    annotation_onset_s,
    annotation_duration_s,
    polarity,
    typical_duration_ms,
    baseline_window_ms,
):
    baseline, noise = _baseline(signal, sfreq, stim_s, baseline_window_ms)
    n = len(signal)
    if annotation_duration_s > 0:
        i0 = max(0, int(round(annotation_onset_s * sfreq)))
        i1 = min(n - 1, max(i0 + 1, int(round((annotation_onset_s + annotation_duration_s) * sfreq))))
        inferred = False
    else:
        anchor = int(round(annotation_onset_s * sfreq))
        radius = max(2, int(round(max(typical_duration_ms, 1.0) / 1000 * sfreq)))
        search0, search1 = max(0, anchor - radius), min(n - 1, anchor + radius)
        oriented = polarity * (signal[search0 : search1 + 1] - baseline)
        peak_i = search0 + int(np.argmax(oriented))
        context0, context1 = search0, search1
        context = polarity * (signal[context0 : context1 + 1] - baseline)
        local_peak = peak_i - context0
        _, _, left_ip, right_ip = _peak_shape(context, local_peak)
        i0 = max(context0, int(np.floor(context0 + left_ip)))
        i1 = min(context1, int(np.ceil(context0 + right_ip)))
        if i1 <= i0:
            half = max(1, int(round(typical_duration_ms / 2000 * sfreq)))
            i0, i1 = max(0, peak_i - half), min(n - 1, peak_i + half)
        inferred = True

    segment = signal[i0 : i1 + 1] - baseline
    oriented = polarity * segment
    local_peak = int(np.argmax(oriented))
    peak_i = i0 + local_peak
    prominence, width_samples, _, _ = _peak_shape(oriented, local_peak)
    times_ms = np.arange(i0, i1 + 1) / sfreq * 1000
    start_latency_ms = (i0 / sfreq - stim_s) * 1000
    end_latency_ms = (i1 / sfreq - stim_s) * 1000
    peak_latency_ms = (peak_i / sfreq - stim_s) * 1000
    return {
        "start_s": i0 / sfreq,
        "end_s": i1 / sfreq,
        "start_latency_ms": start_latency_ms,
        "peak_latency_ms": peak_latency_ms,
        "end_latency_ms": end_latency_ms,
        "duration_ms": end_latency_ms - start_latency_ms,
        "peak_width_ms": width_samples / sfreq * 1000,
        "min_amplitude": float(np.min(segment)),
        "max_amplitude": float(np.max(segment)),
        "signed_peak_amplitude": float(segment[local_peak]),
        "absolute_peak_amplitude": float(np.max(np.abs(segment))),
        "p2p_amplitude": float(np.ptp(segment)),
        "prominence": prominence,
        "auc_abs": float(np.trapezoid(np.abs(segment), times_ms)) if len(segment) >= 2 else 0.0,
        "rms": float(np.sqrt(np.mean(segment**2))),
        "baseline_median": baseline,
        "baseline_noise": noise,
        "amplitude_noise_ratio": float(np.max(np.abs(segment)) / noise),
        "p2p_noise_ratio": float(np.ptp(segment) / noise),
        "prominence_noise_ratio": float(prominence / noise),
        "bounds_inferred": inferred,
    }


def extract_annotation_metrics(
    raw,
    annotations,
    channel="GM Right",
    label_map=None,
    stimulus_label="Stimulus_Auto",
    baseline_window_ms=(-25.0, -5.0),
):
    """Measure manual or reviewed annotations relative to their stimuli."""
    label_map = label_map or {label: label for label in COMPONENTS}
    descriptions = np.asarray(annotations.description, dtype=str)
    stimulus_times_s = np.sort(np.asarray(annotations.onset)[descriptions == stimulus_label])
    if not len(stimulus_times_s):
        raise ValueError(f"No {stimulus_label!r} annotations found")
    selected = []
    for onset, duration, description in zip(annotations.onset, annotations.duration, descriptions):
        if description in label_map:
            selected.append(
                {
                    "onset_s": float(onset),
                    "duration_s": float(duration),
                    "source_label": description,
                    "label": label_map[description],
                }
            )
    annotation_df = pd.DataFrame(selected)
    if annotation_df.empty:
        raise ValueError(f"No annotations found for labels {sorted(label_map)}")
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    polarities = _dominant_polarities(
        annotation_df, signal, sfreq, stimulus_times_s, baseline_window_ms
    )
    typical_durations = _typical_durations(annotation_df)
    rows = []
    for source_index, annotation in annotation_df.iterrows():
        match = _nearest_stimulus(annotation.onset_s, stimulus_times_s)
        if match is None:
            continue
        stimulus_index, stimulus_s = match
        measured = _measure_wave(
            signal,
            sfreq,
            stimulus_s,
            annotation.onset_s,
            annotation.duration_s,
            polarities[annotation.label],
            typical_durations[annotation.label],
            baseline_window_ms,
        )
        rows.append(
            {
                "source_index": int(source_index),
                "source_label": annotation.source_label,
                "label": annotation.label,
                "stimulus_index": int(stimulus_index),
                "stimulus_s": stimulus_s,
                "annotation_onset_s": annotation.onset_s,
                "annotation_duration_ms": annotation.duration_s * 1000,
                "polarity": polarities[annotation.label],
                **measured,
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["stimulus_index", "peak_latency_ms", "label"]
    ).reset_index(drop=True)
    metrics["component_rank"] = (
        metrics.groupby(["stimulus_index", "label"]).cumcount() + 1
    )

    metrics["included_in_calibration"] = True
    metrics["exclusion_reason"] = ""
    for label in COMPONENTS:
        idx = metrics.index[metrics["label"] == label]
        for column in ["peak_latency_ms", "duration_ms"]:
            # LR latency is intentionally multimodal because one stimulus may have
            # several time-ordered LR peaks; do not reject later LR ranks here.
            if label == "lr" and column == "peak_latency_ms":
                continue
            center, scale = robust_center_scale(metrics.loc[idx, column])
            minimum_tolerance = 3.0 if column == "peak_latency_ms" else 2.0
            tolerance = max(3 * scale, minimum_tolerance)
            bad = idx[np.abs(metrics.loc[idx, column] - center) > tolerance]
            metrics.loc[bad, "included_in_calibration"] = False
            reason = f"{column}_outlier"
            metrics.loc[bad, "exclusion_reason"] = metrics.loc[bad, "exclusion_reason"].apply(
                lambda current: f"{current};{reason}".strip(";")
            )

    first_peaks = metrics.pivot_table(
        index="stimulus_index", columns="label", values="peak_latency_ms", aggfunc="first"
    )
    metrics["er_to_mr_ms"] = np.nan
    metrics["mr_to_lr_ms"] = np.nan
    if {"er", "mr"}.issubset(first_peaks.columns):
        mr_rows = metrics["label"] == "mr"
        metrics.loc[mr_rows, "er_to_mr_ms"] = (
            metrics.loc[mr_rows, "peak_latency_ms"]
            - metrics.loc[mr_rows, "stimulus_index"].map(first_peaks["er"])
        )
    if "mr" in first_peaks.columns:
        lr_rows = metrics["label"] == "lr"
        metrics.loc[lr_rows, "mr_to_lr_ms"] = (
            metrics.loc[lr_rows, "peak_latency_ms"]
            - metrics.loc[lr_rows, "stimulus_index"].map(first_peaks["mr"])
        )
    return metrics, polarities, stimulus_times_s


CALIBRATION_METRICS = (
    "start_latency_ms",
    "peak_latency_ms",
    "end_latency_ms",
    "duration_ms",
    "peak_width_ms",
    "absolute_peak_amplitude",
    "p2p_amplitude",
    "prominence",
    "auc_abs",
    "rms",
    "amplitude_noise_ratio",
    "p2p_noise_ratio",
    "prominence_noise_ratio",
)


def build_calibration(metrics, polarities):
    """Build median/MAD detector parameters and a flat summary table."""
    calibration = {
        "method": "median_plus_3_scaled_mad",
        "components": {},
        "sequence": {},
        # Hard significance gates prevent latency-matched baseline fluctuations
        # from being promoted to a complete response. Thresholds remain tied to
        # the manually measured medians; MR is stricter because it is the large,
        # defining middle response in the reference morphology.
        "eligibility": {
            "minimum_median_fraction": {"er": 0.20, "mr": 0.25, "lr": 0.20},
            "maximum_robust_upper_multiplier": 1.50,
        },
    }
    summary_rows = []
    included = metrics[metrics["included_in_calibration"]].copy()
    for label in COMPONENTS:
        group = included[included["label"] == label]
        component = {"polarity": int(polarities[label]), "n": int(len(group)), "metrics": {}}
        for column in CALIBRATION_METRICS:
            center, scale = robust_center_scale(group[column])
            lower, upper = center - 3 * scale, center + 3 * scale
            if column not in {"start_latency_ms", "peak_latency_ms", "end_latency_ms"}:
                lower = max(0.0, lower)
            component["metrics"][column] = {
                "median": center,
                "scaled_mad": scale,
                "lower": lower,
                "upper": upper,
            }
            summary_rows.append(
                {
                    "label": label,
                    "metric": column,
                    "n": len(group),
                    "median": center,
                    "scaled_mad": scale,
                    "lower": lower,
                    "upper": upper,
                }
            )
        calibration["components"][label] = component
        if label == "lr" and len(group):
            # Keep the median as the scoring target but span all manually observed
            # LR ranks in the permissive search window.
            params = component["metrics"]["peak_latency_ms"]
            params["lower"] = min(params["lower"], float(group["peak_latency_ms"].min()))
            params["upper"] = max(params["upper"], float(group["peak_latency_ms"].max()))
    for gap in ["er_to_mr_ms", "mr_to_lr_ms"]:
        if gap in included:
            gap_values = included[gap].dropna()
            center, scale = robust_center_scale(gap_values)
            calibration["sequence"][gap] = {
                "median": center,
                "scaled_mad": scale,
                "lower": max(0.0, center - 3 * scale),
                "upper": center + 3 * scale,
            }
            if gap == "mr_to_lr_ms" and len(gap_values):
                calibration["sequence"][gap]["lower"] = min(
                    calibration["sequence"][gap]["lower"], float(gap_values.min())
                )
                calibration["sequence"][gap]["upper"] = max(
                    calibration["sequence"][gap]["upper"], float(gap_values.max())
                )
    calibration["max_lr_per_stimulus"] = min(
        3, int(metrics[metrics["label"] == "lr"].groupby("stimulus_index").size().max())
    )
    summary = pd.DataFrame(summary_rows)
    for label in COMPONENTS:
        for metric, params in calibration["components"][label]["metrics"].items():
            mask = (summary["label"] == label) & (summary["metric"] == metric)
            summary.loc[mask, ["lower", "upper"]] = [params["lower"], params["upper"]]
    return calibration, summary


def save_calibration(calibration, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, allow_nan=False)


def plot_metric_distributions(metrics, calibration, output_path=None):
    specs = [
        ("peak_latency_ms", "Peak latency", "ms"),
        ("duration_ms", "Duration", "ms"),
        ("absolute_peak_amplitude", "Absolute amplitude", "native"),
        ("p2p_amplitude", "Peak-to-peak", "native"),
        ("prominence", "Prominence", "native"),
        ("peak_width_ms", "Half-prominence width", "ms"),
        ("auc_abs", "Absolute AUC", "native·ms"),
        ("rms", "RMS", "native"),
        ("p2p_noise_ratio", "P2P / baseline noise", "ratio"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (column, title, unit) in zip(axes.ravel(), specs):
        for label, color in zip(COMPONENTS, ["tab:green", "tab:orange", "tab:blue"]):
            values = metrics.loc[
                (metrics["label"] == label) & metrics["included_in_calibration"], column
            ].dropna()
            ax.hist(values, bins=min(20, max(5, len(values))), alpha=0.35, label=label, color=color)
            params = calibration["components"][label]["metrics"][column]
            ax.axvline(params["median"], color=color, linewidth=1.5)
            ax.axvline(params["lower"], color=color, linestyle=":", linewidth=0.9)
            ax.axvline(params["upper"], color=color, linestyle=":", linewidth=0.9)
        ax.set(title=title, xlabel=unit, ylabel="count")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(title="Manual label")
    fig.suptitle("Manual ER/MR/LR calibration distributions", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def _candidate_metrics(signal, sfreq, stim_s, peak_i, polarity, baseline, noise, left_i, right_i):
    segment = signal[left_i : right_i + 1] - baseline
    local_peak = peak_i - left_i
    oriented = polarity * segment
    prominence, width_samples, _, _ = _peak_shape(oriented, local_peak)
    width = float(width_samples / sfreq * 1000)
    t = np.arange(left_i, right_i + 1) / sfreq * 1000
    return {
        "start_s": left_i / sfreq,
        "end_s": right_i / sfreq,
        "start_latency_ms": (left_i / sfreq - stim_s) * 1000,
        "peak_latency_ms": (peak_i / sfreq - stim_s) * 1000,
        "end_latency_ms": (right_i / sfreq - stim_s) * 1000,
        "duration_ms": (right_i - left_i) / sfreq * 1000,
        "peak_width_ms": width,
        "min_amplitude": float(np.min(segment)),
        "max_amplitude": float(np.max(segment)),
        "signed_peak_amplitude": float(segment[local_peak]),
        "absolute_peak_amplitude": float(np.max(np.abs(segment))),
        "p2p_amplitude": float(np.ptp(segment)),
        "prominence": prominence,
        "auc_abs": float(np.trapezoid(np.abs(segment), t)) if len(segment) >= 2 else 0.0,
        "rms": float(np.sqrt(np.mean(segment**2))),
        "baseline_median": baseline,
        "baseline_noise": noise,
        "amplitude_noise_ratio": float(np.max(np.abs(segment)) / noise),
        "p2p_noise_ratio": float(np.ptp(segment) / noise),
        "prominence_noise_ratio": float(prominence / noise),
    }


def _score_candidate(candidate, component):
    score_fields = [
        "peak_latency_ms",
        "duration_ms",
        "peak_width_ms",
        "p2p_amplitude",
        "prominence",
        "p2p_noise_ratio",
    ]
    distances = []
    for field in score_fields:
        params = component["metrics"][field]
        distances.append(abs(candidate[field] - params["median"]) / max(params["scaled_mad"], 1e-12))
    return float(np.mean(np.clip(distances, 0, 10)))


def _component_candidates(
    signal,
    sfreq,
    stim_s,
    label,
    component,
    baseline_window_ms,
    eligibility=None,
):
    polarity = int(component["polarity"])
    latency = component["metrics"]["peak_latency_ms"]
    i0 = max(0, int(round((stim_s + latency["lower"] / 1000) * sfreq)))
    i1 = min(len(signal) - 1, int(round((stim_s + latency["upper"] / 1000) * sfreq)))
    if i1 <= i0 + 2:
        return []
    baseline, noise = _baseline(signal, sfreq, stim_s, baseline_window_ms)
    oriented = polarity * (signal[i0 : i1 + 1] - baseline)
    width_ms = component["metrics"]["peak_width_ms"]
    min_distance = max(1, int(round(max(width_ms["median"] / 2, 0.5) / 1000 * sfreq)))
    local_peaks, props = find_peaks(oriented, prominence=0, distance=min_distance)
    output = []
    duration_cap = max(component["metrics"]["duration_ms"]["upper"], 1.0)
    max_radius = max(2, int(round(duration_cap / 1000 * sfreq)))
    for rank, local_peak in enumerate(local_peaks):
        peak_i = i0 + int(local_peak)
        _, _, left_ip, right_ip = _peak_shape(oriented, int(local_peak))
        left_i = max(i0, int(np.floor(i0 + left_ip)))
        right_i = min(i1, int(np.ceil(i0 + right_ip)))
        left_i = max(left_i, peak_i - max_radius)
        right_i = min(right_i, peak_i + max_radius)
        if right_i <= left_i:
            continue
        candidate = _candidate_metrics(
            signal, sfreq, stim_s, peak_i, polarity, baseline, noise, left_i, right_i
        )
        candidate.update({"label": label, "polarity": polarity, "candidate_rank": rank + 1})
        candidate["score"] = _score_candidate(candidate, component)
        if candidate["score"] > 6.0:
            continue
        if eligibility:
            fraction = float(eligibility["minimum_median_fraction"][label])
            upper_multiplier = float(eligibility["maximum_robust_upper_multiplier"])
            p2p = component["metrics"]["p2p_amplitude"]
            prominence = component["metrics"]["prominence"]
            significant = (
                candidate["p2p_amplitude"] >= fraction * p2p["median"]
                and candidate["prominence"] >= fraction * prominence["median"]
                and candidate["p2p_amplitude"] <= upper_multiplier * p2p["upper"]
                and candidate["prominence"] <= upper_multiplier * prominence["upper"]
            )
            if label == "mr":
                amplitude = component["metrics"]["absolute_peak_amplitude"]
                significant &= candidate["absolute_peak_amplitude"] >= fraction * amplitude["median"]
            if not significant:
                continue
        output.append(candidate)
    return sorted(output, key=lambda row: row["score"])


def detect_components(
    raw,
    stimulus_times_s,
    calibration,
    channel="GM Right",
    baseline_window_ms=(-25.0, -5.0),
):
    """Detect the best ER→MR→1–3 LR sequence after each stimulus."""
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    records = []
    max_lr = int(calibration.get("max_lr_per_stimulus", 3))
    for stimulus_index, stim_s in enumerate(stimulus_times_s):
        by_label = {
            label: _component_candidates(
                signal,
                sfreq,
                float(stim_s),
                label,
                calibration["components"][label],
                baseline_window_ms,
                calibration.get("eligibility"),
            )
            for label in COMPONENTS
        }
        if not by_label["er"]:
            continue
        er = by_label["er"][0]
        er_mr_gap = calibration["sequence"].get("er_to_mr_ms", {})
        mr_options = [
            row
            for row in by_label["mr"]
            if row["peak_latency_ms"] > er["peak_latency_ms"]
            and (
                not er_mr_gap
                or er_mr_gap["lower"]
                <= row["peak_latency_ms"] - er["peak_latency_ms"]
                <= er_mr_gap["upper"]
            )
        ]
        if not mr_options:
            continue
        mr = mr_options[0]
        mr_lr_gap = calibration["sequence"].get("mr_to_lr_ms", {})
        lr_options = [
            row
            for row in by_label["lr"]
            if row["peak_latency_ms"] > mr["peak_latency_ms"]
            and (
                not mr_lr_gap
                or mr_lr_gap["lower"]
                <= row["peak_latency_ms"] - mr["peak_latency_ms"]
                <= mr_lr_gap["upper"]
            )
        ]
        lr_options = sorted(lr_options, key=lambda row: row["peak_latency_ms"])
        if not lr_options:
            continue
        selected = [er, mr, *lr_options[:max_lr]]
        lr_rank = 0
        for candidate in selected:
            if candidate["label"] == "lr":
                lr_rank += 1
            record = dict(candidate)
            record.update(
                {
                    "stimulus_index": int(stimulus_index),
                    "stimulus_s": float(stim_s),
                    "component_rank": lr_rank if candidate["label"] == "lr" else 1,
                    "auto_label": AUTO_LABELS[candidate["label"]],
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def candidates_to_annotations(candidates, orig_time=None):
    if candidates.empty:
        return mne.Annotations([], [], [], orig_time=orig_time)
    ordered = candidates.sort_values("start_s")
    return mne.Annotations(
        onset=ordered["start_s"].to_numpy(),
        duration=(ordered["end_s"] - ordered["start_s"]).to_numpy(),
        description=ordered["auto_label"].to_list(),
        orig_time=orig_time,
    )


def review_annotations(raw, source_annotations, candidates, browser_scaling, block=True):
    """Open a review-only copy; the returned annotations contain user edits."""
    descriptions = np.asarray(source_annotations.description, dtype=str)
    keep = (descriptions == "Stimulus_Auto") | np.char.startswith(descriptions, "BAD") | np.char.startswith(descriptions, "EDGE")
    base = source_annotations[keep]
    review_raw = raw.copy()
    review_raw.set_annotations(base + candidates_to_annotations(candidates, base.orig_time))
    first = float(candidates["start_s"].min()) if len(candidates) else 0.0
    review_raw.plot(
        start=max(0.0, first - 0.1),
        duration=5.0,
        n_channels=len(raw.ch_names),
        scalings={"emg": browser_scaling},
        title="Manual-calibrated ER/MR/LR review — edit lowercase *_auto spans",
        block=block,
    )
    return review_raw.annotations


def normalized_review_annotations(raw, review_annotations, channel="GM Right"):
    """Measure reviewed labels and return full-span normalized auto annotations."""
    label_map = {auto: label for label, auto in AUTO_LABELS.items()}
    metrics, _, _ = extract_annotation_metrics(
        raw, review_annotations, channel=channel, label_map=label_map
    )
    metrics["auto_label"] = metrics["label"].map(AUTO_LABELS)
    normalized = candidates_to_annotations(metrics, review_annotations.orig_time)
    descriptions = np.asarray(review_annotations.description, dtype=str)
    keep = (descriptions == "Stimulus_Auto") | np.char.startswith(descriptions, "BAD") | np.char.startswith(descriptions, "EDGE")
    return review_annotations[keep] + normalized, metrics


def atomic_save_annotations(annotations, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp-annot.fif")
    annotations.save(tmp_path, overwrite=True)
    roundtrip = mne.read_annotations(tmp_path)
    expected = pd.Series(annotations.description, dtype=str).value_counts().to_dict()
    actual = pd.Series(roundtrip.description, dtype=str).value_counts().to_dict()
    if len(roundtrip) != len(annotations) or actual != expected:
        raise RuntimeError("Annotation round-trip validation failed")
    os.replace(tmp_path, output_path)
    return actual


def evaluate_candidates(manual_metrics, candidates, calibration):
    """Greedy in-sample peak-latency matching by stimulus and component."""
    rows = []
    manual = manual_metrics[manual_metrics["included_in_calibration"]]
    calibration_stimuli = set(manual_metrics["stimulus_index"].astype(int))
    for label in COMPONENTS:
        tolerance = max(
            2.0,
            3 * calibration["components"][label]["metrics"]["peak_latency_ms"]["scaled_mad"],
        )
        component_manual = manual[manual["label"] == label]
        component_auto = candidates[
            (candidates["label"] == label)
            & candidates["stimulus_index"].astype(int).isin(calibration_stimuli)
        ]
        matched_manual = set()
        matched_auto = set()
        for auto_i, auto in component_auto.iterrows():
            options = component_manual[
                (component_manual["stimulus_index"] == auto["stimulus_index"])
                & (~component_manual.index.isin(matched_manual))
            ]
            if len(options):
                delta = (options["peak_latency_ms"] - auto["peak_latency_ms"]).abs()
                best = delta.idxmin()
                if delta.loc[best] <= tolerance:
                    matched_manual.add(best)
                    matched_auto.add(auto_i)
        tp = len(matched_manual)
        fp = len(component_auto) - tp
        fn = len(component_manual) - tp
        rows.append(
            {
                "label": label,
                "manual": len(component_manual),
                "automatic": len(component_auto),
                "matched": tp,
                "missed": fn,
                "extra": fp,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "match_tolerance_ms": tolerance,
            }
        )
    return pd.DataFrame(rows)


def plot_best_epochs(
    raw,
    reviewed_metrics,
    output_path=None,
    channel="GM Right",
    n_plots=10,
    plot_window_ms=None,
    stimulus_times_s=None,
    time_scale=0.50,
    row_height=2.4,
    right_edge_padding_ms=0.5,
):
    """Plot the best reviewed epochs from each stimulus to the next stimulus."""
    lr = reviewed_metrics[reviewed_metrics["label"] == "lr"]
    ranking = (
        lr.groupby("stimulus_index")
        .agg(lr_prominence=("prominence", "max"), lr_amplitude=("absolute_peak_amplitude", "max"))
        .sort_values(["lr_prominence", "lr_amplitude"], ascending=False)
        .head(n_plots)
        .reset_index()
    )
    if ranking.empty:
        return None, ranking
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if row_height <= 0:
        raise ValueError("row_height must be positive")
    if right_edge_padding_ms < 0:
        raise ValueError("right_edge_padding_ms must be non-negative")
    stimulus_times_s = None if stimulus_times_s is None else np.asarray(stimulus_times_s, dtype=float)
    fig, axes = plt.subplots(
        len(ranking),
        1,
        figsize=(9 * time_scale, max(row_height * len(ranking), 4)),
        sharex=False,
        squeeze=False,
    )
    axes = axes[:, 0]
    colors = {"er": "tab:green", "mr": "tab:orange", "lr": "tab:blue"}
    for plot_index, (_, ranked) in enumerate(ranking.iterrows()):
        ax = axes[plot_index]
        group = reviewed_metrics[reviewed_metrics["stimulus_index"] == ranked.stimulus_index]
        stim_s = float(group["stimulus_s"].iloc[0])
        left_ms = 0.0
        if stimulus_times_s is not None:
            stimulus_index = int(ranked.stimulus_index)
            if stimulus_index < 0 or stimulus_index >= len(stimulus_times_s) - 1:
                raise ValueError(f"Stimulus {stimulus_index} has no following stimulus for full-epoch plotting")
            right_ms = float((stimulus_times_s[stimulus_index + 1] - stimulus_times_s[stimulus_index]) * 1000)
        elif plot_window_ms is not None:
            left_ms, right_ms = map(float, plot_window_ms)
        else:
            raise ValueError("Pass stimulus_times_s or plot_window_ms")
        if right_ms <= left_ms:
            raise ValueError(f"Invalid epoch window: {left_ms}..{right_ms} ms")
        i0 = max(0, int(round((stim_s + left_ms / 1000) * sfreq)))
        i1 = min(len(signal), int(round((stim_s + right_ms / 1000) * sfreq)) + 1)
        x = (np.arange(i0, i1) / sfreq - stim_s) * 1000
        y = signal[i0:i1]
        ax.plot(x, y, color="black", linewidth=1.0)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        for component in COMPONENTS:
            for _, row in group[group["label"] == component].iterrows():
                ax.axvspan(row.start_latency_ms, row.end_latency_ms, color=colors[component], alpha=0.2, label=component.upper() if plot_index == 0 else None)
                ax.axvline(row.peak_latency_ms, color=colors[component], linewidth=0.8)
        # Match the original ER/LR notebook: each subplot is independently
        # autoscaled by Matplotlib. A large response in one epoch therefore does
        # not flatten the smaller responses in the other epochs.
        ax.axvline(right_ms, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlim(left_ms, right_ms + right_edge_padding_ms)
        ax.margins(y=0.05)
        ax.set_title(f"stimulus {int(ranked.stimulus_index)}", loc="left", fontsize=9)
        ax.grid(alpha=0.25, linestyle=":")
    axes[-1].set_xlabel("Time after stimulus, ms")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(f"{channel} Channel Epochs", fontsize=12, weight="bold")
    fig.legend(
        unique.values(), unique.keys(), fontsize=8, loc="upper right",
        ncol=3, bbox_to_anchor=(0.98, 0.965), frameon=False,
    )
    fig.text(
        0.015, 0.5,
        "Amplitude, native",
        fontsize=9, rotation="vertical", va="center",
    )
    fig.tight_layout(rect=[0.04, 0.02, 1, 0.96])
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ranking
