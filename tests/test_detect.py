"""Tests for the detection module."""

import numpy as np
import pytest
from PIL import Image

from app.detect import Detection, _get_severity, run_detection


class TestDetection:
    def test_to_dict_has_required_keys(self):
        det = Detection(
            label="scratch",
            confidence=0.75,
            severity="medium",
            bbox=(0.1, 0.2, 0.5, 0.6),
        )
        result = det.to_dict()
        assert set(result.keys()) == {"label", "confidence", "severity", "bbox"}
        assert result["label"] == "scratch"
        assert result["confidence"] == 0.75

    def test_confidence_rounded_to_3dp(self):
        det = Detection("pit", 0.123456, "low", (0, 0, 1, 1))
        assert det.to_dict()["confidence"] == 0.123


class TestGetSeverity:
    @pytest.mark.parametrize(
        "conf, expected",
        [
            (0.40, "low"),
            (0.59, "low"),
            (0.61, "medium"),
            (0.79, "medium"),
            (0.85, "high"),
            (0.99, "high"),
        ],
    )
    def test_severity_buckets(self, conf, expected):
        assert _get_severity(conf) == expected


class TestRunDetection:
    def test_returns_list(self):
        """run_detection should always return a list (even if empty)."""
        dummy_image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        # Uses placeholder yolov8n — may return 0 or more detections
        result = run_detection(dummy_image)
        assert isinstance(result, list)

    def test_detections_sorted_by_confidence(self):
        """Results should be sorted highest confidence first."""
        dummy_image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        result = run_detection(dummy_image)
        confidences = [d.confidence for d in result]
        assert confidences == sorted(confidences, reverse=True)
