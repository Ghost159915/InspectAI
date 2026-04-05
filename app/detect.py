"""
detect.py — Defect Detection Module
====================================

Wraps the fine-tuned YOLOv8 model and exposes a single public API:
``run_detection(image) -> list[Detection]``.

Model resolution order
----------------------
1. ``models/inspectai_yolov8.pt`` on disk  (local run / Docker)
2. Download from HF Hub repo ``AngryGhostMan/inspectai``  (HF Spaces)
3. ``yolov8n.pt`` COCO fallback            (development without fine-tuned weights)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = Path("models/inspectai_yolov8.pt")

# HF Hub repo that hosts the trained weights.
# Override with the HF_MODEL_REPO environment variable if needed.
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "AngryGhostMan/inspectai")
HF_MODEL_FILE = "inspectai_yolov8.pt"

CONFIDENCE_THRESHOLD = 0.35

SEVERITY_MAP: dict[tuple[float, float], str] = {
    (0.35, 0.60): "low",
    (0.60, 0.80): "medium",
    (0.80, 1.00): "high",
}


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    severity: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalised [0, 1]

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "confidence": round(self.confidence, 3),
            "severity":   self.severity,
            "bbox":       self.bbox,
        }


# ── Model loader ───────────────────────────────────────────────────────────────

_model: YOLO | None = None


def _download_from_hub(dest: Path) -> YOLO:
    """Download weights from HF Hub and return a loaded YOLO instance."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"[detect] Downloading weights from {HF_MODEL_REPO}/{HF_MODEL_FILE} …")
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            local_dir=str(dest.parent),
            token=os.getenv("HF_TOKEN"),
        )
        print(f"[detect] Weights saved → {local}")
        return YOLO(local)
    except Exception as exc:
        print(
            f"[detect] HF Hub download failed: {exc}\n"
            "         Falling back to yolov8n.pt (COCO classes — not defect classes).\n"
            f"         Fix: push inspectai_yolov8.pt to {HF_MODEL_REPO} on HF Hub."
        )
        return YOLO("yolov8n.pt")


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> YOLO:
    """Load and cache the YOLO model for the lifetime of the process."""
    global _model
    if _model is None:
        if model_path.exists():
            print(f"[detect] Loading model from '{model_path}' …")
            _model = YOLO(str(model_path))
        else:
            _model = _download_from_hub(model_path)
    return _model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_severity(confidence: float) -> str:
    for (low, high), label in SEVERITY_MAP.items():
        if low <= confidence < high:
            return label
    return "high"


# ── Public API ─────────────────────────────────────────────────────────────────

def run_detection(
    image: Image.Image | np.ndarray,
    model_path: Path = DEFAULT_MODEL_PATH,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[Detection]:
    """
    Run defect detection on a single image.

    Returns a list of Detection instances sorted by confidence (highest first).
    Returns an empty list for a clean surface.
    """
    model = load_model(model_path)
    results = model(image, conf=conf_threshold, verbose=False)

    detections: list[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        img_h, img_w = result.orig_shape[0], result.orig_shape[1]
        for box in result.boxes:
            conf  = float(box.conf[0])
            label = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)
            detections.append(Detection(
                label=label,
                confidence=conf,
                severity=_get_severity(conf),
                bbox=bbox,
            ))

    return sorted(detections, key=lambda d: d.confidence, reverse=True)
