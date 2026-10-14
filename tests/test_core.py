"""Small, data-independent regression tests for the reusable helpers."""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/emg-lr-segmentation-mne")

from emg_lr_segmentation.manual_calibrated_er_mr_lr import robust_center_scale  # noqa: E402
from emg_lr_segmentation.movement_phase import assign_phase_bins, make_phase_bins  # noqa: E402


class CoreHelperTests(unittest.TestCase):
    def test_robust_center_scale_ignores_nonfinite_values(self):
        center, scale = robust_center_scale([1.0, 2.0, 3.0, np.nan, np.inf])
        self.assertEqual(center, 2.0)
        self.assertGreater(scale, 0.0)

    def test_phase_bins_include_positive_pi(self):
        edges, _ = make_phase_bins(8)
        phases = pd.DataFrame({"movement_phase": [-np.pi, 0.0, np.pi]})
        result = assign_phase_bins(phases, edges)
        self.assertEqual(result["phase_bin"].tolist(), [1, 5, 8])

    def test_tracked_calibration_has_required_metrics(self):
        path = PROJECT_ROOT / "config" / "er_morphology_thresholds.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "er_start_ms",
            "er_peak_ms",
            "er_width_ms",
            "er_full_duration_ms",
            "er_rebound_depth_ratio",
        }
        self.assertTrue(required.issubset(payload["metrics"]))


if __name__ == "__main__":
    unittest.main()
