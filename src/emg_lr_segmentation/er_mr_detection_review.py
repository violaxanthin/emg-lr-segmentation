"""Notebook 4 ER/MR detection and editable review workflow."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .manual_calibrated_er_mr_lr import (
    _component_candidates,
    atomic_save_annotations,
    candidates_to_annotations,
    extract_annotation_metrics,
)

# Keep the calibrated ER/MR detector's original baseline convention.
ER_MR_BASELINE_MS = (-25.0, -5.0)
REVIEW_DURATION_S = 5.0
REVIEW_SCALING = 5.16987885
TEMPLATE_PLOT_COUNT = 10


def detect_er_mr(raw, stimulus_times_s, calibration, channel):
    """Reuse calibrated candidates and ER→MR selection without requiring LR."""
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    records = []
    for stimulus_index, stim_s in enumerate(stimulus_times_s):
        options = {
            label: _component_candidates(
                signal, sfreq, float(stim_s), label,
                calibration["components"][label], ER_MR_BASELINE_MS,
                calibration.get("eligibility"),
            )
            for label in ("er", "mr")
        }
        if not options["er"]:
            continue
        er = options["er"][0]
        gap = calibration["sequence"].get("er_to_mr_ms", {})
        mr_options = [
            row for row in options["mr"]
            if row["peak_latency_ms"] > er["peak_latency_ms"]
            and (not gap or gap["lower"] <= row["peak_latency_ms"] - er["peak_latency_ms"] <= gap["upper"])
        ]
        for candidate in [er, *mr_options[:1]]:
            record = dict(candidate)
            record.update(stimulus_index=stimulus_index, stimulus_s=float(stim_s),
                          component_rank=1, auto_label=f'{candidate["label"]}_auto')
            records.append(record)
    return pd.DataFrame(records, columns=None if records else [
        "stimulus_index", "stimulus_s", "label", "auto_label", "start_s", "end_s",
    ])


def mark_er_mr(raw, source_annotations, stimulus_times_s, channel, project_root):
    """Save new annotations/metrics without touching previous pipeline files."""
    root = Path(project_root)
    calibration_path = root / "outputs/metrics/manual_er_mr_lr_calibration.json"
    with calibration_path.open() as stream:
        calibration = json.load(stream)
    candidates = detect_er_mr(raw, stimulus_times_s, calibration, channel)
    # Original calibration responses stay in their original file, not this browser.
    descriptions = np.asarray(source_annotations.description, dtype=str)
    base = source_annotations[descriptions == "Stimulus_Auto"]
    annotations = base + candidates_to_annotations(candidates, base.orig_time)
    atomic_save_annotations(annotations, root / "outputs/annotations/notebook4_er_mr_auto-annot.fif")
    candidates.to_csv(root / "outputs/metrics/notebook4_er_mr_auto.csv", index=False)
    return candidates, annotations


def _response_rows(annotations):
    """Return sorted ER/MR annotation rows for save verification."""
    rows = [
        (str(description), float(onset), float(duration))
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        )
        if str(description) in {"er_auto", "mr_auto"}
    ]
    return sorted(rows)


def _verify_saved_responses(expected, saved, sfreq):
    """Verify labels and boundaries after the FIF round trip."""
    expected_rows = _response_rows(expected)
    saved_rows = _response_rows(saved)
    if len(expected_rows) != len(saved_rows):
        raise RuntimeError("Saved ER/MR annotation count differs from the reviewed count")
    tolerance_s = 0.5 / float(sfreq)
    for expected_row, saved_row in zip(expected_rows, saved_rows):
        if expected_row[0] != saved_row[0] or not np.allclose(
            expected_row[1:], saved_row[1:], atol=tolerance_s, rtol=0
        ):
            raise RuntimeError("Saved ER/MR labels or boundaries differ from the review")


def keep_complete_er_mr_epochs(annotations):
    """Keep response spans only for stimuli containing both ER and MR."""
    descriptions = np.asarray(annotations.description, dtype=str)
    stimulus_mask = descriptions == "Stimulus_Auto"
    response_mask = np.isin(descriptions, ["er_auto", "mr_auto"])
    stimuli = annotations[stimulus_mask]
    responses = annotations[response_mask]
    stimulus_times_s = np.asarray(stimuli.onset, dtype=float)

    response_stimulus_indices = np.searchsorted(
        stimulus_times_s, np.asarray(responses.onset, dtype=float), side="right"
    ) - 1
    labels_by_stimulus = {}
    for stimulus_index, label in zip(response_stimulus_indices, responses.description):
        if stimulus_index >= 0:
            labels_by_stimulus.setdefault(int(stimulus_index), set()).add(str(label))
    complete_indices = {
        stimulus_index
        for stimulus_index, labels in labels_by_stimulus.items()
        if {"er_auto", "mr_auto"}.issubset(labels)
    }
    keep_response = np.asarray(
        [int(index) in complete_indices for index in response_stimulus_indices],
        dtype=bool,
    )
    complete = stimuli + responses[keep_response]

    status = []
    for stimulus_index, stimulus_time_s in enumerate(stimulus_times_s):
        labels = labels_by_stimulus.get(stimulus_index, set())
        status.append(
            {
                "stimulus_index": stimulus_index,
                "stimulus_time_s": stimulus_time_s,
                "er_count": sum(
                    index == stimulus_index and label == "er_auto"
                    for index, label in zip(response_stimulus_indices, responses.description)
                ),
                "mr_count": sum(
                    index == stimulus_index and label == "mr_auto"
                    for index, label in zip(response_stimulus_indices, responses.description)
                ),
                "complete_er_mr": stimulus_index in complete_indices,
            }
        )
    return complete, pd.DataFrame(status)


def keep_mr_and_paired_er(annotations):
    """Keep every MR, but retain ER only when the same stimulus has an MR."""
    descriptions = np.asarray(annotations.description, dtype=str)
    stimuli = annotations[descriptions == "Stimulus_Auto"]
    responses = annotations[np.isin(descriptions, ["er_auto", "mr_auto"])]
    stimulus_times_s = np.asarray(stimuli.onset, dtype=float)
    response_stimulus_indices = np.searchsorted(
        stimulus_times_s, np.asarray(responses.onset, dtype=float), side="right"
    ) - 1
    mr_stimulus_indices = {
        int(stimulus_index)
        for stimulus_index, label in zip(
            response_stimulus_indices, responses.description
        )
        if stimulus_index >= 0 and str(label) == "mr_auto"
    }
    keep_response = np.asarray(
        [
            str(label) == "mr_auto" or int(stimulus_index) in mr_stimulus_indices
            for stimulus_index, label in zip(
                response_stimulus_indices, responses.description
            )
        ],
        dtype=bool,
    )
    return stimuli + responses[keep_response]


def review_er_mr(raw, auto_annotations, channel, project_root, start_s=0):
    """Edit responses, verify persistence, and retain complete ER+MR epochs."""
    root = Path(project_root)
    output_path = root / "outputs/annotations/notebook4_er_mr_reviewed-annot.fif"
    complete_path = root / "outputs/annotations/notebook4_er_mr_complete-annot.fif"
    metrics_path = root / "outputs/metrics/notebook4_er_mr_reviewed.csv"
    status_path = root / "outputs/metrics/notebook4_er_mr_epoch_status.csv"
    response_labels = {"er_auto", "mr_auto"}
    previous = mne.read_annotations(output_path) if output_path.exists() else auto_annotations
    previous = keep_mr_and_paired_er(previous)
    previous_responses = previous[np.isin(previous.description, list(response_labels))]
    stimuli = auto_annotations[np.asarray(auto_annotations.description) == "Stimulus_Auto"]
    review_raw = raw.copy()
    review_raw.set_annotations(stimuli + previous_responses)
    print("Edit, add, or delete er_auto/mr_auto spans. Close the window to save.")
    mne.viz.set_browser_backend("qt")
    review_raw.plot(start=start_s, duration=REVIEW_DURATION_S,
                   scalings={"emg": REVIEW_SCALING}, block=True,
                   title="Editable ER/MR review")
    edited = review_raw.annotations
    responses = edited[np.isin(edited.description, list(response_labels))]
    # Preserve all MR edits, but never save ER in an epoch without MR.
    reviewed = keep_mr_and_paired_er(stimuli + responses)
    atomic_save_annotations(reviewed, output_path)
    saved_reviewed = mne.read_annotations(output_path)
    _verify_saved_responses(reviewed, saved_reviewed, raw.info["sfreq"])

    complete, epoch_status = keep_complete_er_mr_epochs(saved_reviewed)
    metrics, _, _ = extract_annotation_metrics(
        raw,
        complete,
        channel=channel,
        label_map={"er_auto": "er", "mr_auto": "mr"},
        baseline_window_ms=ER_MR_BASELINE_MS,
    )
    atomic_save_annotations(complete, complete_path)
    metrics.to_csv(metrics_path, index=False)
    epoch_status.to_csv(status_path, index=False)
    return saved_reviewed, complete, metrics, epoch_status


def _baseline_center(signal, sfreq, stimulus_time_s):
    """Return the median from the final 5 ms before a stimulus."""
    end = int(round(stimulus_time_s * sfreq))
    start = int(round((stimulus_time_s - 0.005) * sfreq))
    if start < 0 or end <= start:
        raise ValueError(f"Invalid 5 ms baseline before stimulus {stimulus_time_s:.6f} s")
    return float(np.median(signal[start:end]))


def build_er_mr_template(
    raw,
    complete_annotations,
    channel,
    project_root,
):
    """Build a stimulus-aligned masked median through each epoch's MR end."""
    root = Path(project_root)
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(complete_annotations.description, dtype=str)
    stimulus_times_s = np.asarray(
        complete_annotations.onset[descriptions == "Stimulus_Auto"], dtype=float
    )
    response_mask = np.isin(descriptions, ["er_auto", "mr_auto"])
    response_onsets = np.asarray(complete_annotations.onset[response_mask], dtype=float)
    response_durations = np.asarray(
        complete_annotations.duration[response_mask], dtype=float
    )
    response_labels = descriptions[response_mask]
    response_indices = np.searchsorted(
        stimulus_times_s, response_onsets, side="right"
    ) - 1
    epoch_rows_by_stimulus = {}
    for stimulus_index in sorted(set(response_indices.astype(int))):
        epoch_rows = response_indices == stimulus_index
        if {"er_auto", "mr_auto"}.issubset(set(response_labels[epoch_rows])):
            epoch_rows_by_stimulus[stimulus_index] = epoch_rows
    source_indices = np.asarray(sorted(epoch_rows_by_stimulus), dtype=int)
    if not len(source_indices):
        raise ValueError("No epochs containing both ER and MR are available")

    source_centers = []
    source_mr_end_samples = []
    source_er_start_samples = []
    for stimulus_index in source_indices:
        epoch_rows = epoch_rows_by_stimulus[stimulus_index]
        stimulus_time_s = stimulus_times_s[stimulus_index]
        er_start_s = float(
            np.min(response_onsets[epoch_rows & (response_labels == "er_auto")])
        )
        mr_rows = epoch_rows & (response_labels == "mr_auto")
        mr_end_s = float(np.max(response_onsets[mr_rows] + response_durations[mr_rows]))
        er_start_sample = max(
            0, int(round((er_start_s - stimulus_time_s) * sfreq))
        )
        mr_end_sample = int(round((mr_end_s - stimulus_time_s) * sfreq))
        if mr_end_sample <= 0:
            raise ValueError(
                f"MR ends at or before the stimulus for stimulus {stimulus_index}"
            )
        source_er_start_samples.append(er_start_sample)
        source_mr_end_samples.append(mr_end_sample)
        source_centers.append(_baseline_center(signal, sfreq, stimulus_time_s))

    support_samples = int(max(source_mr_end_samples))
    source_waveforms = np.full(
        (len(source_indices), support_samples), np.nan, dtype=float
    )
    source_mask = np.zeros_like(source_waveforms, dtype=bool)
    for row_index, (stimulus_index, baseline_center, mr_end_sample) in enumerate(
        zip(source_indices, source_centers, source_mr_end_samples)
    ):
        start = int(round(stimulus_times_s[stimulus_index] * sfreq))
        available = min(mr_end_sample, len(signal) - start)
        if available != mr_end_sample:
            raise ValueError(
                f"MR boundary exceeds recording for stimulus {stimulus_index}"
            )
        source_waveforms[row_index, :mr_end_sample] = (
            signal[start : start + mr_end_sample] - baseline_center
        )
        source_mask[row_index, :mr_end_sample] = True
    template = np.nanmedian(source_waveforms, axis=0)
    contributor_count = source_mask.sum(axis=0)
    time_ms = np.arange(support_samples) / sfreq * 1000

    fit_rows = []
    for row_index, stimulus_index in enumerate(source_indices):
        stimulus_time_s = stimulus_times_s[stimulus_index]
        fit_start = source_er_start_samples[row_index]
        fit_end = source_mr_end_samples[row_index]
        start = int(round(stimulus_time_s * sfreq))
        baseline_center = source_centers[row_index]
        epoch = signal[start : start + fit_end] - baseline_center
        valid = fit_end > fit_start
        reason = ""
        scale = np.nan
        if valid:
            template_fit = template[fit_start:fit_end]
            denominator = float(np.dot(template_fit, template_fit))
            if denominator <= np.finfo(float).eps:
                valid = False
                reason = "zero_template_energy"
            else:
                scale = float(
                    np.dot(epoch[fit_start:fit_end], template_fit) / denominator
                )
        else:
            reason = "invalid_er_mr_fit_interval"
        fit_rows.append(
            {
                "stimulus_index": stimulus_index,
                "stimulus_time_s": stimulus_time_s,
                "baseline_center": baseline_center,
                "template_scale": scale,
                "fit_start_ms": fit_start / sfreq * 1000,
                "fit_end_ms": fit_end / sfreq * 1000,
                "fit_valid": valid,
                "fit_reason": reason,
            }
        )
    fits = pd.DataFrame(fit_rows)
    response_metrics, _, _ = extract_annotation_metrics(
        raw,
        complete_annotations,
        channel=channel,
        label_map={"er_auto": "er", "mr_auto": "mr"},
        baseline_window_ms=ER_MR_BASELINE_MS,
    )
    peak_latencies = (
        response_metrics
        .pivot_table(
            index="stimulus_index",
            columns="label",
            values="peak_latency_ms",
            aggfunc="first",
        )
        .rename(columns={"er": "er_peak_ms", "mr": "mr_peak_ms"})
        .reset_index()
    )
    fits = fits.merge(peak_latencies, on="stimulus_index", how="left")
    output_dir = root / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "notebook4_er_mr_template.npz",
        source_indices=source_indices,
        source_waveforms=source_waveforms,
        source_mask=source_mask,
        source_mr_end_samples=np.asarray(source_mr_end_samples),
        source_baseline_centers=np.asarray(source_centers),
        template=template,
        contributor_count=contributor_count,
        time_ms=time_ms,
        sfreq=sfreq,
    )
    fits.to_csv(output_dir / "notebook4_er_mr_template_fits.csv", index=False)
    return {
        "source_indices": source_indices,
        "source_waveforms": source_waveforms,
        "source_mask": source_mask,
        "source_mr_end_samples": np.asarray(source_mr_end_samples),
        "source_baseline_centers": np.asarray(source_centers),
        "template": template,
        "contributor_count": contributor_count,
        "time_ms": time_ms,
        "fits": fits,
    }


def plot_er_mr_template(template_result, raw, channel):
    """Plot source epochs, their median template, and representative fits."""
    time_ms = template_result["time_ms"]
    source_waveforms = template_result["source_waveforms"]
    template = template_result["template"]
    source_indices = template_result["source_indices"]
    fig_sources, ax = plt.subplots(figsize=(11, 4))
    for stimulus_index, waveform in zip(source_indices, source_waveforms):
        ax.plot(time_ms, waveform, color="0.65", linewidth=0.8, alpha=0.7)
    ax.plot(time_ms, template, color="tab:red", linewidth=2, label="median template")
    ax.set(
        title=f"Masked median ER+MR template ({len(source_indices)} complete epochs)",
        xlabel="Time after stimulus, ms",
        ylabel="Amplitude, native units",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig_sources.tight_layout()

    fits = template_result["fits"]
    fits = fits.loc[fits["fit_valid"]].head(TEMPLATE_PLOT_COUNT)
    if fits.empty:
        return fig_sources, None
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    fig_fits, axes = plt.subplots(
        len(fits), 1, figsize=(11, max(2.0 * len(fits), 4)),
        sharex=True, squeeze=False,
    )
    for ax, row in zip(axes[:, 0], fits.itertuples()):
        start = int(round(row.stimulus_time_s * sfreq))
        available = min(int(round(row.fit_end_ms / 1000 * sfreq)), len(signal) - start)
        epoch = signal[start : start + available] - row.baseline_center
        ax.plot(time_ms[:available], epoch, color="black", linewidth=0.8, label="epoch")
        ax.plot(
            time_ms[:available],
            row.template_scale * template[:available],
            color="tab:red",
            linewidth=1.2,
            label="fitted template",
        )
        for column, label, color, marker in (
            ("er_peak_ms", "ER peak", "tab:red", "o"),
            ("mr_peak_ms", "MR peak", "tab:blue", "o"),
        ):
            peak_ms = getattr(row, column, np.nan)
            peak_index = (
                int(round((row.stimulus_time_s + peak_ms / 1000) * sfreq))
                if np.isfinite(peak_ms) else -1
            )
            end_tolerance_ms = 500 / sfreq
            if (
                0 <= peak_index < len(signal)
                and peak_ms <= row.fit_end_ms + end_tolerance_ms
            ):
                ax.scatter(
                    peak_ms,
                    signal[peak_index] - row.baseline_center,
                    s=38,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=5,
                    label=label,
                )
        ax.axvspan(row.fit_start_ms, row.fit_end_ms, color="tab:blue", alpha=0.1)
        ax.set_title(
            f"stimulus {row.stimulus_index} | scale {row.template_scale:.3g}",
            loc="left", fontsize=9,
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time after stimulus, ms")
    fig_fits.text(0.02, 0.5, "Amplitude, native units", rotation="vertical", va="center")
    fig_fits.tight_layout(rect=[0.04, 0, 1, 1])
    return fig_sources, fig_fits
