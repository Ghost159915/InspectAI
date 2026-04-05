"""Tests for the agent module (mocked LLM — no Ollama required for CI)."""

import json
from unittest.mock import patch

import pytest

from app.agent import _extract_json, generate_report
from app.detect import Detection

SAMPLE_DETECTIONS = [
    Detection("scratch", 0.82, "high", (0.1, 0.1, 0.4, 0.3)),
    Detection("pit", 0.55, "low", (0.6, 0.6, 0.8, 0.8)),
]

MOCK_REPORT = {
    "timestamp": "2026-01-01T00:00:00",
    "overall_status": "FAIL",
    "defect_count": 2,
    "summary": "High-severity scratch detected. Immediate review required.",
    "defects": [
        {
            "label": "scratch",
            "severity": "high",
            "confidence": 0.82,
            "recommended_action": "Remove from line; escalate to engineering.",
        },
        {
            "label": "pit",
            "severity": "low",
            "confidence": 0.55,
            "recommended_action": "Monitor; re-inspect at next scheduled cycle.",
        },
    ],
    "notes": "",
}


class TestGenerateReport:
    def test_empty_detections_returns_pass(self):
        report = generate_report([])
        assert report["overall_status"] == "PASS"
        assert report["defect_count"] == 0

    @patch("app.agent._get_llm")
    @patch("app.agent.retrieve_context", return_value="mocked context")
    def test_report_structure(self, mock_rag, mock_llm):
        mock_llm.return_value.invoke.return_value = json.dumps(MOCK_REPORT)
        report = generate_report(SAMPLE_DETECTIONS)
        assert "overall_status" in report
        assert "defects" in report
        assert isinstance(report["defects"], list)

    @patch("app.agent._get_llm")
    @patch("app.agent.retrieve_context", return_value="mocked context")
    def test_raises_on_invalid_json(self, mock_rag, mock_llm):
        mock_llm.return_value.invoke.return_value = "not valid json at all"
        with pytest.raises(ValueError, match="invalid JSON"):
            generate_report(SAMPLE_DETECTIONS)


class TestExtractJson:
    def test_strips_markdown_fences(self):
        wrapped = '```json\n{"key": "value"}\n```'
        assert _extract_json(wrapped) == '{"key": "value"}'

    def test_plain_json_unchanged(self):
        plain = '{"key": "value"}'
        assert _extract_json(plain) == plain
