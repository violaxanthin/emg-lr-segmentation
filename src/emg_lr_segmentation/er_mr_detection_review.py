"""Notebook 4 ER/MR detection and editable review workflow."""
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .manual_calibrated_er_mr_lr import (
    _baseline,
    _candidate_metrics,
    _component_candidates,
    _peak_shape,
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


def _response_rows(annotations, labels=None):
    """Return sorted response annotation rows for save verification."""
    labels = {"er_auto", "mr_auto"} if labels is None else set(labels)
    rows = [
        (str(description), float(onset), float(duration))
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        )
        if str(description) in labels
    ]
    return sorted(rows)


def _verify_saved_responses(expected, saved, sfreq, labels=None):
    """Verify labels and boundaries after the FIF round trip."""
    expected_rows = _response_rows(expected, labels)
    saved_rows = _response_rows(saved, labels)
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


def find_lr_candidates(
    raw,
    complete_annotations,
    channel,
    search_config,
    project_root,
):
    """Find every calibrated-polarity peak between MR end and the LR window end."""
    start_offset_ms = float(search_config["start_after_mr_end_ms"])
    search_end_ms = float(search_config["end_after_stimulus_ms"])
    p2p_min = float(search_config["p2p_min_native"])
    if start_offset_ms < 0 or search_end_ms <= 0 or p2p_min < 0:
        raise ValueError("LR search-window limits must be positive")

    root = Path(project_root)
    with (root / "outputs/metrics/manual_er_mr_lr_calibration.json").open() as stream:
        calibration = json.load(stream)
    lr_component = calibration["components"]["lr"]
    polarity = int(lr_component["polarity"])

    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(complete_annotations.description, dtype=str)
    stimulus_times_s = np.asarray(
        complete_annotations.onset[descriptions == "Stimulus_Auto"], dtype=float
    )
    mr_mask = descriptions == "mr_auto"
    mr_onsets = np.asarray(complete_annotations.onset[mr_mask], dtype=float)
    mr_ends = mr_onsets + np.asarray(
        complete_annotations.duration[mr_mask], dtype=float
    )
    mr_stimulus_indices = np.searchsorted(
        stimulus_times_s, mr_onsets, side="right"
    ) - 1

    er_mask = descriptions == "er_auto"
    er_onsets = np.asarray(complete_annotations.onset[er_mask], dtype=float)
    er_ends = er_onsets + np.asarray(
        complete_annotations.duration[er_mask], dtype=float
    )
    er_stimulus_indices = np.searchsorted(
        stimulus_times_s, er_onsets, side="right"
    ) - 1
    invalid_order = []
    for stimulus_index in sorted(set(mr_stimulus_indices.astype(int))):
        epoch_er_ends = er_ends[er_stimulus_indices == stimulus_index]
        epoch_mr_onsets = mr_onsets[mr_stimulus_indices == stimulus_index]
        if (
            not len(epoch_er_ends)
            or not len(epoch_mr_onsets)
            or float(np.max(epoch_er_ends)) > float(np.min(epoch_mr_onsets))
        ):
            invalid_order.append(int(stimulus_index))
    if invalid_order:
        raise ValueError(
            "ER must end before MR starts before searching for LR candidates; "
            f"fix stimulus indices {invalid_order}"
        )

    records = []
    for stimulus_index in sorted(set(mr_stimulus_indices.astype(int))):
        if stimulus_index < 0:
            continue
        stimulus_time_s = float(stimulus_times_s[stimulus_index])
        epoch_mr_ends = mr_ends[mr_stimulus_indices == stimulus_index]
        mr_end_s = float(np.max(epoch_mr_ends))
        search_start_s = mr_end_s + start_offset_ms / 1000
        search_stop_s = stimulus_time_s + search_end_ms / 1000
        i0 = max(0, int(np.ceil(search_start_s * sfreq)))
        i1 = min(len(signal) - 1, int(np.floor(search_stop_s * sfreq)))
        if i1 <= i0 + 2:
            continue

        baseline, noise = _baseline(
            signal, sfreq, stimulus_time_s, ER_MR_BASELINE_MS
        )
        oriented = polarity * (signal[i0 : i1 + 1] - baseline)
        local_peaks, _ = find_peaks(
            oriented,
            prominence=0,
        )
        for candidate_rank, local_peak in enumerate(local_peaks, start=1):
            peak_i = i0 + int(local_peak)
            _, _, left_ip, right_ip = _peak_shape(oriented, int(local_peak))
            left_i = max(i0, int(np.floor(i0 + left_ip)))
            right_i = min(i1, int(np.ceil(i0 + right_ip)))
            if right_i <= left_i:
                continue
            candidate = _candidate_metrics(
                signal,
                sfreq,
                stimulus_time_s,
                peak_i,
                polarity,
                baseline,
                noise,
                left_i,
                right_i,
            )
            candidate.update(
                {
                    "stimulus_index": stimulus_index,
                    "stimulus_time_s": stimulus_time_s,
                    "label": "lr",
                    "candidate_rank": candidate_rank,
                    "polarity": polarity,
                    "mr_end_latency_ms": (mr_end_s - stimulus_time_s) * 1000,
                    "search_start_ms": (i0 / sfreq - stimulus_time_s) * 1000,
                    "search_end_ms": (i1 / sfreq - stimulus_time_s) * 1000,
                    "p2p_min_native": p2p_min,
                    "p2p_pass": candidate["p2p_amplitude"] >= p2p_min,
                    "auto_label": "lr_auto",
                }
            )
            records.append(candidate)

    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["stimulus_index", "peak_latency_ms"], ignore_index=True
        )
        candidates["candidate_rank"] = (
            candidates.groupby("stimulus_index").cumcount() + 1
        )
    output_dir = root / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "notebook4_lr_candidates.csv", index=False)
    return candidates


def review_lr_and_tune_search(
    raw,
    complete_annotations,
    candidates,
    channel,
    project_root,
    start_s=0,
):
    """Review P2P-accepted LR spans and summarize the reviewed search limits."""
    root = Path(project_root)
    auto_path = root / "outputs/annotations/notebook4_lr_auto-annot.fif"
    reviewed_path = root / "outputs/annotations/notebook4_lr_reviewed-annot.fif"
    metrics_path = root / "outputs/metrics/notebook4_lr_reviewed.csv"
    tuning_path = root / "outputs/metrics/notebook4_lr_search_tuning.csv"

    accepted = candidates.loc[candidates["p2p_pass"]].copy()
    auto_lr = candidates_to_annotations(accepted, complete_annotations.orig_time)
    automatic = complete_annotations + auto_lr
    atomic_save_annotations(automatic, auto_path)

    if reviewed_path.exists():
        previous = mne.read_annotations(reviewed_path)
        previous_lr = previous[
            np.asarray(previous.description, dtype=str) == "lr_auto"
        ]
    else:
        previous_lr = auto_lr
    review_raw = raw.copy()
    review_raw.set_annotations(complete_annotations + previous_lr)
    print("Edit, add, split, merge, or delete lr_auto spans. Close the window to save.")
    mne.viz.set_browser_backend("qt")
    review_raw.plot(
        start=start_s,
        duration=REVIEW_DURATION_S,
        scalings={"emg": REVIEW_SCALING},
        block=True,
        title="Editable LR review and search tuning",
    )

    edited = review_raw.annotations
    reviewed_lr = edited[np.asarray(edited.description, dtype=str) == "lr_auto"]
    reviewed = complete_annotations + reviewed_lr
    atomic_save_annotations(reviewed, reviewed_path)
    saved = mne.read_annotations(reviewed_path)
    _verify_saved_responses(
        reviewed,
        saved,
        raw.info["sfreq"],
        labels={"lr_auto"},
    )

    if not len(reviewed_lr):
        metrics = pd.DataFrame()
        tuning = pd.DataFrame()
    else:
        metrics, _, _ = extract_annotation_metrics(
            raw,
            saved,
            channel=channel,
            label_map={"lr_auto": "lr"},
            baseline_window_ms=ER_MR_BASELINE_MS,
        )
        descriptions = np.asarray(complete_annotations.description, dtype=str)
        stimulus_times_s = np.asarray(
            complete_annotations.onset[descriptions == "Stimulus_Auto"], dtype=float
        )
        mr_mask = descriptions == "mr_auto"
        mr_onsets = np.asarray(complete_annotations.onset[mr_mask], dtype=float)
        mr_ends = mr_onsets + np.asarray(
            complete_annotations.duration[mr_mask], dtype=float
        )
        mr_indices = np.searchsorted(stimulus_times_s, mr_onsets, side="right") - 1
        mr_end_ms = {
            int(index): float(np.max(mr_ends[mr_indices == index]) - stimulus_times_s[index]) * 1000
            for index in set(mr_indices.astype(int))
            if index >= 0
        }
        metrics["mr_end_latency_ms"] = metrics["stimulus_index"].map(mr_end_ms)
        metrics["start_after_mr_end_ms"] = (
            metrics["start_latency_ms"] - metrics["mr_end_latency_ms"]
        )
        tuning = metrics[[
            "start_after_mr_end_ms",
            "start_latency_ms",
            "peak_latency_ms",
            "end_latency_ms",
            "p2p_amplitude",
            "duration_ms",
        ]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    metrics.to_csv(metrics_path, index=False)
    tuning.to_csv(tuning_path)
    return saved, metrics, tuning

