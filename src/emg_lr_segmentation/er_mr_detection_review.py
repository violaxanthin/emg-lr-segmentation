"""Notebook 4 ER/MR detection and editable review workflow."""
import copy
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .manual_calibrated_er_mr_lr import (
    CALIBRATION_METRICS,
    _baseline,
    _candidate_metrics,
    _component_candidates,
    _peak_shape,
    atomic_save_annotations,
    candidates_to_annotations,
    extract_annotation_metrics,
    robust_center_scale,
)

# Keep the calibrated ER/MR detector's original baseline convention.
ER_MR_BASELINE_MS = (-25.0, -5.0)
REVIEW_DURATION_S = 5.0
# Display-only MNE browser range: approximately +/- 5.2 native signal units.
# This does not change the recorded data or any detector threshold.
EMG_REVIEW_DISPLAY_RANGE_NATIVE = 5.2


def detect_er_mr(raw, stimulus_times_s, calibration, channel):
    """Detect ER/MR with strict raw-signal polarity and ER-before-MR order."""
    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    records = []
    for stimulus_index, stim_s in enumerate(stimulus_times_s):
        options = {}
        for label in ("er", "mr"):
            candidates = _component_candidates(
                signal, sfreq, float(stim_s), label,
                calibration["components"][label], ER_MR_BASELINE_MS,
                calibration.get("eligibility"),
            )
            expected_raw_sign = 1 if label == "er" else -1
            for candidate in candidates:
                raw_peak_amplitude = (
                    candidate["baseline_median"]
                    + candidate["signed_peak_amplitude"]
                )
                candidate["raw_peak_amplitude"] = raw_peak_amplitude
                candidate["absolute_polarity_pass"] = bool(
                    expected_raw_sign * raw_peak_amplitude > 0
                )
            options[label] = [
                candidate
                for candidate in candidates
                if candidate["absolute_polarity_pass"]
            ]
        if not options["er"]:
            continue
        er = options["er"][0]
        # ER and MR are first detected from their own fixed calibration windows.
        # The sequence is applied only afterwards: MR requires an earlier ER.
        mr_options = [
            row for row in options["mr"]
            if row["start_latency_ms"] >= er["end_latency_ms"]
            and row["raw_peak_amplitude"] < er["raw_peak_amplitude"]
        ]
        for candidate in [er, *mr_options[:1]]:
            record = dict(candidate)
            record.update(stimulus_index=stimulus_index, stimulus_s=float(stim_s),
                          component_rank=1, auto_label=f'{candidate["label"]}_auto')
            records.append(record)
    return pd.DataFrame(records, columns=None if records else [
        "stimulus_index", "stimulus_s", "label", "auto_label", "start_s", "end_s",
    ])


def retune_er_mr_calibration(
    reviewed_metrics,
    base_calibration,
    lower_quantile=0.05,
    upper_quantile=0.95,
):
    """Retune ER/MR from reviewed spans while preserving the LR calibration.

    Medians and scaled MADs remain the candidate-scoring targets. Empirical
    quantiles define the search ranges, and high-amplitude responses are not
    rejected merely for exceeding a symmetric upper amplitude bound.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Calibration quantiles must satisfy 0 <= lower < upper <= 1")
    required = {"label", "polarity", "stimulus_index", *CALIBRATION_METRICS}
    missing = required - set(reviewed_metrics.columns)
    if missing:
        raise ValueError(f"Reviewed ER/MR metrics are missing columns: {sorted(missing)}")

    calibration = copy.deepcopy(base_calibration)
    summary_rows = []
    for label in ("er", "mr"):
        group = reviewed_metrics.loc[reviewed_metrics["label"].eq(label)].copy()
        if group.empty:
            raise ValueError(f"No reviewed {label.upper()} metrics are available")
        polarity = int(group["polarity"].mode().iloc[0])
        component = {"polarity": polarity, "n": int(len(group)), "metrics": {}}
        for metric in CALIBRATION_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"No finite {label.upper()} values for {metric}")
            median, scaled_mad = robust_center_scale(values)
            lower = float(values.quantile(lower_quantile))
            upper = float(values.quantile(upper_quantile))
            if metric not in {"start_latency_ms", "peak_latency_ms", "end_latency_ms"}:
                lower = max(0.0, lower)
            component["metrics"][metric] = {
                "median": median,
                "scaled_mad": scaled_mad,
                "lower": lower,
                "upper": upper,
            }
            summary_rows.append({
                "label": label,
                "metric": metric,
                "n": len(values),
                "median": median,
                "scaled_mad": scaled_mad,
                "lower": lower,
                "upper": upper,
            })
        calibration["components"][label] = component

    peak_latencies = reviewed_metrics.pivot_table(
        index="stimulus_index",
        columns="label",
        values="peak_latency_ms",
        aggfunc="first",
    ).dropna(subset=["er", "mr"])
    er_to_mr = peak_latencies["mr"] - peak_latencies["er"]
    if er_to_mr.empty:
        raise ValueError("No complete reviewed ER/MR pairs are available")
    gap_median, gap_scale = robust_center_scale(er_to_mr)
    calibration.setdefault("sequence", {})["er_to_mr_ms"] = {
        "median": gap_median,
        "scaled_mad": gap_scale,
        "lower": max(0.0, float(er_to_mr.quantile(lower_quantile))),
        "upper": float(er_to_mr.quantile(upper_quantile)),
    }

    eligibility = calibration.setdefault("eligibility", {})
    eligibility.setdefault("minimum_median_fraction", {"er": 0.20, "mr": 0.25, "lr": 0.20})
    # Large genuine responses are common in the reviewed data. Keep only the
    # lower significance gates for this retuned calibration.
    eligibility["maximum_robust_upper_multiplier"] = None
    calibration["er_mr_retuning"] = {
        "method": "reviewed_median_scaled_mad_with_empirical_quantile_ranges",
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "complete_pair_count": int(len(peak_latencies)),
    }
    return calibration, pd.DataFrame(summary_rows)


def mark_er_mr(
    raw,
    source_annotations,
    stimulus_times_s,
    channel,
    project_root,
    calibration_path=None,
):
    """Save new annotations/metrics without touching previous pipeline files."""
    root = Path(project_root)
    calibration_path = (
        Path(calibration_path)
        if calibration_path is not None
        else root / "outputs/metrics/manual_er_mr_lr_calibration.json"
    )
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


def keep_er_and_paired_mr(annotations):
    """Keep every ER, but retain MR only when the same stimulus has an ER."""
    descriptions = np.asarray(annotations.description, dtype=str)
    stimuli = annotations[descriptions == "Stimulus_Auto"]
    responses = annotations[np.isin(descriptions, ["er_auto", "mr_auto"])]
    stimulus_times_s = np.asarray(stimuli.onset, dtype=float)
    response_stimulus_indices = np.searchsorted(
        stimulus_times_s, np.asarray(responses.onset, dtype=float), side="right"
    ) - 1
    er_stimulus_indices = {
        int(stimulus_index)
        for stimulus_index, label in zip(
            response_stimulus_indices, responses.description
        )
        if stimulus_index >= 0 and str(label) == "er_auto"
    }
    keep_response = np.asarray(
        [
            str(label) == "er_auto" or int(stimulus_index) in er_stimulus_indices
            for stimulus_index, label in zip(
                response_stimulus_indices, responses.description
            )
        ],
        dtype=bool,
    )
    return stimuli + responses[keep_response]


def keep_mr_and_paired_er(annotations):
    """Backward-compatible alias for the corrected ER-first constraint."""
    return keep_er_and_paired_mr(annotations)


def keep_lr_in_complete_er_mr_epochs(complete_annotations, lr_annotations):
    """Retain LR spans only in stimulus epochs containing both ER and MR."""
    normalized_complete, epoch_status = keep_complete_er_mr_epochs(complete_annotations)
    complete_indices = set(
        epoch_status.loc[epoch_status["complete_er_mr"], "stimulus_index"].astype(int)
    )
    descriptions = np.asarray(normalized_complete.description, dtype=str)
    stimulus_times_s = np.asarray(
        normalized_complete.onset[descriptions == "Stimulus_Auto"], dtype=float
    )
    lr_descriptions = np.asarray(lr_annotations.description, dtype=str)
    lr_only = lr_annotations[lr_descriptions == "lr_auto"]
    if not len(lr_only) or not len(stimulus_times_s):
        return lr_only
    lr_stimulus_indices = np.searchsorted(
        stimulus_times_s, np.asarray(lr_only.onset, dtype=float), side="right"
    ) - 1
    keep = np.asarray(
        [int(index) in complete_indices for index in lr_stimulus_indices],
        dtype=bool,
    )
    return lr_only[keep]


def review_er_mr(
    raw,
    auto_annotations,
    channel,
    project_root,
    start_s=0,
    seed_from_saved_review=True,
):
    """Edit responses, verify persistence, and retain complete ER+MR epochs."""
    root = Path(project_root)
    output_path = root / "outputs/annotations/notebook4_er_mr_reviewed-annot.fif"
    complete_path = root / "outputs/annotations/notebook4_er_mr_complete-annot.fif"
    metrics_path = root / "outputs/metrics/notebook4_er_mr_reviewed.csv"
    status_path = root / "outputs/metrics/notebook4_er_mr_epoch_status.csv"
    response_labels = {"er_auto", "mr_auto"}
    previous = (
        mne.read_annotations(output_path)
        if seed_from_saved_review and output_path.exists()
        else auto_annotations
    )
    previous = keep_er_and_paired_mr(previous)
    previous_responses = previous[np.isin(previous.description, list(response_labels))]
    stimuli = auto_annotations[np.asarray(auto_annotations.description) == "Stimulus_Auto"]
    review_raw = raw.copy()
    review_raw.set_annotations(stimuli + previous_responses)
    print("Edit, add, or delete er_auto/mr_auto spans. Close the window to save.")
    mne.viz.set_browser_backend("qt")
    review_raw.plot(start=start_s, duration=REVIEW_DURATION_S,
                   scalings={"emg": EMG_REVIEW_DISPLAY_RANGE_NATIVE}, block=True,
                   title="Editable ER/MR review")
    edited = review_raw.annotations
    responses = edited[np.isin(edited.description, list(response_labels))]
    # ER may exist alone, but MR is valid only when the same epoch contains ER.
    reviewed = keep_er_and_paired_mr(stimuli + responses)
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
    """Detect duration-capped LR from the MR right edge in complete epochs."""
    start_offset_ms = float(search_config["start_after_mr_end_ms"])
    search_end_ms = float(search_config["end_after_stimulus_ms"])
    p2p_min = float(search_config["p2p_min_native"])
    if start_offset_ms < 0 or search_end_ms <= 0 or p2p_min < 0:
        raise ValueError(
            "LR start offset and P2P minimum must be non-negative; search end "
            "must be positive"
        )

    root = Path(project_root)
    configured_max_duration = search_config.get("max_duration_ms")
    if configured_max_duration is None:
        manual_metrics_path = root / "outputs/metrics/manual_er_mr_lr_metrics.csv"
        if not manual_metrics_path.exists():
            raise FileNotFoundError(manual_metrics_path)
        manual_metrics = pd.read_csv(manual_metrics_path)
        manual_lr_durations = pd.to_numeric(
            manual_metrics.loc[manual_metrics["label"].eq("lr"), "duration_ms"],
            errors="coerce",
        ).dropna()
        if manual_lr_durations.empty:
            raise ValueError("No manually annotated LR durations are available")
        max_duration_ms = float(manual_lr_durations.max())
    else:
        max_duration_ms = float(configured_max_duration)
    if max_duration_ms <= 0:
        raise ValueError("LR maximum duration must be positive")

    with (root / "outputs/metrics/manual_er_mr_lr_calibration.json").open() as stream:
        calibration = json.load(stream)
    lr_component = calibration["components"]["lr"]
    polarity = int(lr_component["polarity"])
    lr_latency = lr_component["metrics"]["peak_latency_ms"]
    calibration_peak_lower_ms = float(lr_latency["lower"])
    stable_end_ms = min(float(lr_latency["upper"]), search_end_ms)
    if stable_end_ms <= 0:
        raise ValueError("The LR search end is not positive")

    complete_annotations, epoch_status = keep_complete_er_mr_epochs(
        complete_annotations
    )
    complete_indices = set(
        epoch_status.loc[epoch_status["complete_er_mr"], "stimulus_index"].astype(int)
    )

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
    for stimulus_index in sorted(complete_indices):
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

    mr_end_by_stimulus = {
        int(stimulus_index): float(np.max(mr_ends[mr_stimulus_indices == stimulus_index]))
        for stimulus_index in complete_indices
    }

    records = []
    max_duration_samples = max(
        2, int(np.floor(max_duration_ms / 1000 * sfreq))
    )
    for stimulus_index in sorted(complete_indices):
        stimulus_time_s = float(stimulus_times_s[stimulus_index])
        mr_end_s = mr_end_by_stimulus[stimulus_index]
        # LR is allowed to begin at the first sample on or after the MR right
        # boundary; there is deliberately no positive ER/MR-to-LR dead zone.
        search_start_s = mr_end_s + start_offset_ms / 1000
        search_stop_s = stimulus_time_s + stable_end_ms / 1000
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
            raw_duration_samples = right_i - left_i
            duration_was_capped = raw_duration_samples > max_duration_samples
            if duration_was_capped:
                left_extent = peak_i - left_i
                right_extent = right_i - peak_i
                total_extent = left_extent + right_extent
                left_keep = int(round(max_duration_samples * left_extent / total_extent))
                left_keep = min(left_extent, max(1, left_keep))
                right_keep = min(
                    right_extent,
                    max(1, max_duration_samples - left_keep),
                )
                remaining = max_duration_samples - left_keep - right_keep
                if remaining > 0:
                    add_left = min(remaining, left_extent - left_keep)
                    left_keep += add_left
                    remaining -= add_left
                    right_keep += min(remaining, right_extent - right_keep)
                left_i = peak_i - left_keep
                right_i = peak_i + right_keep
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
                    "calibration_peak_latency_lower_ms": calibration_peak_lower_ms,
                    "stable_search_end_ms": stable_end_ms,
                    "actual_search_start_ms": (i0 / sfreq - stimulus_time_s) * 1000,
                    "lr_max_duration_ms": max_duration_ms,
                    "duration_was_capped": duration_was_capped,
                    "p2p_min_native": p2p_min,
                    "p2p_pass": candidate["p2p_amplitude"] >= p2p_min,
                    "auto_label": "lr_auto",
                }
            )
            records.append(candidate)

    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates["mr_end_latency_ms"] = candidates["stimulus_index"].map(
            {
                index: (mr_end_s - stimulus_times_s[index]) * 1000
                for index, mr_end_s in mr_end_by_stimulus.items()
            }
        )
        candidates["start_after_mr_end_ms"] = (
            candidates["start_latency_ms"] - candidates["mr_end_latency_ms"]
        )
        candidates = candidates.loc[
            candidates["start_after_mr_end_ms"] >= start_offset_ms
        ].copy()
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
    seed_from_saved_review=True,
):
    """Review LR spans from complete ER+MR epochs and summarize search limits."""
    root = Path(project_root)
    auto_path = root / "outputs/annotations/notebook4_lr_auto-annot.fif"
    reviewed_path = root / "outputs/annotations/notebook4_lr_reviewed-annot.fif"
    metrics_path = root / "outputs/metrics/notebook4_lr_reviewed.csv"
    tuning_path = root / "outputs/metrics/notebook4_lr_search_tuning.csv"

    complete_annotations, epoch_status = keep_complete_er_mr_epochs(
        complete_annotations
    )
    complete_indices = set(
        epoch_status.loc[epoch_status["complete_er_mr"], "stimulus_index"].astype(int)
    )
    accepted = candidates.loc[
        candidates["p2p_pass"]
        & candidates["stimulus_index"].astype(int).isin(complete_indices)
    ].copy()
    auto_lr = candidates_to_annotations(accepted, complete_annotations.orig_time)
    automatic = complete_annotations + auto_lr
    atomic_save_annotations(automatic, auto_path)

    if seed_from_saved_review and reviewed_path.exists():
        previous = mne.read_annotations(reviewed_path)
        previous_lr_count = int(
            np.sum(np.asarray(previous.description, dtype=str) == "lr_auto")
        )
        previous_lr = keep_lr_in_complete_er_mr_epochs(
            complete_annotations, previous
        )
        removed_previous_lr = previous_lr_count - len(previous_lr)
        if removed_previous_lr:
            print(
                "Excluded saved LR spans outside current complete ER+MR epochs:",
                removed_previous_lr,
            )
        descriptions = np.asarray(complete_annotations.description, dtype=str)
        stimulus_times_s = np.asarray(
            complete_annotations.onset[descriptions == "Stimulus_Auto"], dtype=float
        )
        previous_indices = set(
            np.searchsorted(
                stimulus_times_s,
                np.asarray(previous_lr.onset, dtype=float),
                side="right",
            ).astype(int)
            - 1
        )
        auto_indices = np.searchsorted(
            stimulus_times_s,
            np.asarray(auto_lr.onset, dtype=float),
            side="right",
        ) - 1
        auto_for_unreviewed_epochs = auto_lr[
            np.asarray(
                [int(index) not in previous_indices for index in auto_indices],
                dtype=bool,
            )
        ]
        review_lr_seed = previous_lr + auto_for_unreviewed_epochs
        print(
            "LR review seed: saved valid =",
            len(previous_lr),
            "; auto in not-yet-reviewed complete epochs =",
            len(auto_for_unreviewed_epochs),
        )
    else:
        review_lr_seed = auto_lr
    review_raw = raw.copy()
    review_raw.set_annotations(complete_annotations + review_lr_seed)
    print("Edit, add, split, merge, or delete lr_auto spans. Close the window to save.")
    mne.viz.set_browser_backend("qt")
    review_raw.plot(
        start=start_s,
        duration=REVIEW_DURATION_S,
        scalings={"emg": EMG_REVIEW_DISPLAY_RANGE_NATIVE},
        block=True,
        title="Editable LR review and search tuning",
    )

    edited = review_raw.annotations
    edited_lr_count = int(
        np.sum(np.asarray(edited.description, dtype=str) == "lr_auto")
    )
    reviewed_lr = keep_lr_in_complete_er_mr_epochs(
        complete_annotations, edited
    )
    removed_edited_lr = edited_lr_count - len(reviewed_lr)
    if removed_edited_lr:
        print(
            "Excluded edited LR spans outside current complete ER+MR epochs:",
            removed_edited_lr,
        )
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


def calculate_lr_to_mr_p2p_ratio(
    raw,
    complete_annotations,
    reviewed_annotations,
    reviewed_lr_metrics,
    channel,
    project_root,
):
    """Calculate LR P2P / reviewed-MR P2P on the original analysis signal."""
    if reviewed_lr_metrics.empty:
        raise ValueError("No reviewed LR metrics are available")

    signal = raw.get_data(picks=[channel])[0]
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(complete_annotations.description, dtype=str)
    stimulus_times_s = np.asarray(
        complete_annotations.onset[descriptions == "Stimulus_Auto"], dtype=float
    )
    mr_mask = descriptions == "mr_auto"
    mr_onsets = np.asarray(complete_annotations.onset[mr_mask], dtype=float)
    mr_durations = np.asarray(complete_annotations.duration[mr_mask], dtype=float)
    mr_indices = np.searchsorted(stimulus_times_s, mr_onsets, side="right") - 1

    mr_p2p_by_stimulus = {}
    for stimulus_index in sorted(set(mr_indices.astype(int))):
        rows = np.flatnonzero(mr_indices == stimulus_index)
        if stimulus_index < 0 or len(rows) != 1:
            continue
        row = int(rows[0])
        start = max(0, int(round(mr_onsets[row] * sfreq)))
        end = min(
            len(signal),
            int(round((mr_onsets[row] + mr_durations[row]) * sfreq)),
        )
        if end > start:
            mr_p2p_by_stimulus[int(stimulus_index)] = float(
                np.ptp(signal[start:end])
            )

    reviewed_descriptions = np.asarray(reviewed_annotations.description, dtype=str)
    lr_mask = reviewed_descriptions == "lr_auto"
    lr_onsets = np.asarray(reviewed_annotations.onset[lr_mask], dtype=float)
    lr_durations = np.asarray(reviewed_annotations.duration[lr_mask], dtype=float)
    ratio_rows = []
    unavailable_stimuli = set()
    for source_index, (onset, duration) in enumerate(zip(lr_onsets, lr_durations)):
        stimulus_index = int(np.searchsorted(stimulus_times_s, onset, side="right") - 1)
        start = max(0, int(round(onset * sfreq)))
        end = min(len(signal), int(round((onset + duration) * sfreq)))
        mr_p2p = mr_p2p_by_stimulus.get(stimulus_index, np.nan)
        lr_p2p = float(np.ptp(signal[start:end])) if end > start else np.nan
        ratio = (
            lr_p2p / mr_p2p
            if np.isfinite(lr_p2p)
            and np.isfinite(mr_p2p)
            and mr_p2p > np.finfo(float).eps
            else np.nan
        )
        if not np.isfinite(ratio):
            unavailable_stimuli.add(stimulus_index)
        ratio_rows.append(
            {
                "source_index": source_index,
                "mr_p2p_amplitude": mr_p2p,
                "lr_p2p_amplitude": lr_p2p,
                "lr_to_mr_p2p_ratio": ratio,
            }
        )

    enriched = reviewed_lr_metrics.merge(
        pd.DataFrame(ratio_rows),
        on="source_index",
        how="left",
        validate="one_to_one",
    )
    if unavailable_stimuli:
        print(
            "LR/MR P2P ratio unavailable for stimulus indices:",
            sorted(unavailable_stimuli),
        )
    output_dir = Path(project_root) / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "notebook4_lr_mr_p2p_ratio_metrics.csv", index=False)
    return enriched

