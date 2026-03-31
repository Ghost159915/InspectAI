"""
detect.py — Defect Detection Module
====================================

This module is the first stage of the InspectAI pipeline. It wraps a
fine-tuned YOLOv8 model and exposes a single clean API: ``run_detection()``.

Pipeline position:
    Image Input → [detect.py] → agent.py → rag.py → ui.py

How it works:
    1. ``load_model()`` initialises the YOLOv8 model on first call and caches
       it in the module-level ``_model`` variable. Subsequent calls return the
       cached instance — no repeated disk I/O.
    2. ``run_detection()`` passes the image through YOLO inference, filters
       results below ``conf_threshold``, and converts raw YOLO output into
       typed ``Detection`` dataclass instances.
    3. Each detection is assigned a human-readable severity level (low /
       medium / high) based on which confidence bucket it falls into. These
       buckets mirror common industrial defect grading standards.
    4. Results are returned sorted by confidence (highest first) so the agent
       can prioritise the most certain findings.

Model notes:
    - Default weights: ``models/inspectai_yolov8.pt`` (fine-tuned on MVTec AD)
    - Fallback: if weights are missing, ``yolov8n.pt`` (COCO pretrained) is
      used automatically. This keeps the pipeline runnable during development
      before training is complete; detections will be object classes (person,
      car, …) rather than defect classes.
    - Bounding boxes are returned as normalised floats [0, 1] so they are
      resolution-independent and can be rescaled by the UI layer.

Dependencies:
    ultralytics — YOLOv8 inference engine
    Pillow      — Image I/O and format compatibility
    numpy       — Array support for direct numpy image inputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the fine-tuned YOLOv8 weights file.
# Populated after running notebooks/training.ipynb.
# If absent, the pipeline falls back to yolov8n.pt (see load_model).
DEFAULT_MODEL_PATH = Path("models/inspectai_yolov8.pt")

# Detections below this confidence are discarded entirely.
# 0.35 is a conservative floor — keeps recall high for safety-critical
# inspection use cases while filtering pure noise.
CONFIDENCE_THRESHOLD = 0.35

# Severity bucket thresholds.
# Key: (lower_bound_inclusive, upper_bound_exclusive)
# Value: severity label used in reports and UI colour coding.
#
# Rationale for the 0.60 / 0.80 split:
#   - 0.35–0.60: model is uncertain; likely a real defect but operator
#                should verify before escalating → "low" / monitor
#   - 0.60–0.80: model is reasonably confident; warrants review → "medium"
#   - 0.80–1.00: model is highly confident; warrants immediate action → "high"
SEVERITY_MAP: dict[tuple[float, float], str] = {
    (0.35, 0.60): "low",
    (0.60, 0.80): "medium",
    (0.80, 1.00): "high",
}


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Represents a single defect detected in an image.

    This is the primary data transfer object (DTO) passed between the
    detection, agent, and UI layers.

    Attributes:
        label:      Defect class name as defined in the YOLO dataset config
                    (e.g. "scratch", "pit", "crack", "contamination").
        confidence: Raw model confidence score in range [0.0, 1.0]. Higher
                    values indicate the model is more certain this is a real
                    defect of this class.
        severity:   Human-readable severity bucket ("low" / "medium" / "high")
                    derived from confidence via SEVERITY_MAP. Used by the agent
                    to select appropriate recommended actions and by the UI for
                    colour coding bounding boxes.
        bbox:       Normalised bounding box as (x1, y1, x2, y2) where all
                    values are in [0, 1] relative to image width/height.
                    Normalisation makes bboxes resolution-independent.
                    The UI layer scales these to pixel coordinates for drawing.
    """

    label: str
    confidence: float
    severity: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalised

    def to_dict(self) -> dict:
        """
        Serialise the detection to a plain dict for JSON embedding in prompts.

        Confidence is rounded to 3 decimal places to keep prompts concise and
        avoid floating-point noise confusing the LLM.

        Returns:
            Dict with keys: label, confidence, severity, bbox.
        """
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "bbox": self.bbox,
        }


# ── Model Loader ──────────────────────────────────────────────────────────────

# Module-level cache. None until first call to load_model().
# Using a module-level variable (rather than a class attribute or lru_cache)
# keeps the code simple and compatible with Gradio's multi-process launch.
_model: YOLO | None = None


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Load the YOLOv8 model and cache it for the lifetime of the process.

    On first call the model weights are read from disk and loaded into memory.
    All subsequent calls return the already-loaded instance immediately,
    avoiding repeated I/O on every inference request.

    Fallback behaviour:
        If ``model_path`` does not exist (e.g. fine-tuning not yet run),
        ``yolov8n.pt`` is downloaded automatically by Ultralytics on first
        use. This allows the full pipeline to run during development — the
        detected class names will be COCO categories rather than defect types.

    Args:
        model_path: Path to the YOLOv8 ``.pt`` weights file.

    Returns:
        Loaded and ready ``YOLO`` instance.
    """
    global _model
    if _model is None:
        if not model_path.exists():
            # Development fallback — warn clearly so the developer knows
            # they are not using the production defect-detection weights.
            print(
                f"[detect] WARNING: Model not found at '{model_path}'.\n"
                "         Falling back to yolov8n.pt (COCO classes, not defect classes).\n"
                "         Run notebooks/training.ipynb to produce the fine-tuned weights."
            )
            _model = YOLO("yolov8n.pt")
        else:
            print(f"[detect] Loading model from '{model_path}'...")
            _model = YOLO(str(model_path))
    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_severity(confidence: float) -> str:
    """
    Map a raw confidence score to a severity label using SEVERITY_MAP.

    Iterates through the (lower, upper) → label mapping. The final bucket
    (0.80–1.00) is open-ended — any confidence >= 0.80 returns "high".

    Args:
        confidence: Model confidence in [0.0, 1.0].

    Returns:
        Severity label: "low", "medium", or "high".
    """
    for (low, high), label in SEVERITY_MAP.items():
        if low <= confidence < high:
            return label
    # Catches confidence == 1.0 (exact upper bound) and any floating-point
    # edge cases above 0.80 that miss the last bucket.
    return "high"


# ── Public API ────────────────────────────────────────────────────────────────

def run_detection(
    image: Image.Image | np.ndarray,
    model_path: Path = DEFAULT_MODEL_PATH,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[Detection]:
    """
    Run defect detection on a single image and return structured results.

    This is the primary entry point for the detection stage. It is called
    directly by both the Gradio UI (``ui.py``) and the test suite.

    Processing steps:
        1. Load (or retrieve cached) YOLO model.
        2. Run inference with the specified confidence threshold.
           Ultralytics handles NMS (non-maximum suppression) internally.
        3. For each detection, extract class name, confidence, and bbox.
        4. Normalise pixel-space bbox coordinates to [0, 1].
        5. Assign severity via ``_get_severity()``.
        6. Sort results by confidence descending so the agent sees the
           most certain defects first in its prompt context.

    Args:
        image:          Input image as a PIL Image (any mode) or a numpy array
                        in (H, W, C) RGB layout. Both formats are accepted
                        natively by Ultralytics.
        model_path:     Path to the YOLOv8 weights to use.
        conf_threshold: Minimum confidence for a detection to be kept.
                        Lower values increase recall; higher values increase
                        precision. Default 0.35 is appropriate for inspection
                        tasks where missing a defect is worse than a false alarm.

    Returns:
        List of ``Detection`` instances sorted by confidence (highest first).
        Returns an empty list if no defects meet the threshold — this is a
        valid "clean surface" result and is handled gracefully by ``agent.py``.

    Example::

        from PIL import Image
        from app.detect import run_detection

        img = Image.open("data/samples/scratch_example.jpg")
        detections = run_detection(img)
        for d in detections:
            print(f"{d.label} ({d.severity}) — {d.confidence:.1%}")
    """
    model = load_model(model_path)

    # verbose=False suppresses Ultralytics' per-frame console output which
    # clutters the Gradio logs during interactive use.
    results = model(image, conf=conf_threshold, verbose=False)

    detections: list[Detection] = []

    for result in results:
        # result.boxes is None if YOLO found no objects above threshold.
        if result.boxes is None:
            continue

        # Image dimensions used for bbox normalisation.
        # result.orig_shape is (height, width) — note the axis order.
        img_h, img_w = result.orig_shape[0], result.orig_shape[1]

        for box in result.boxes:
            conf = float(box.conf[0])
            # Class index → human-readable label from the dataset YAML config.
            label = result.names[int(box.cls[0])]
            # box.xyxy[0] gives absolute pixel coordinates (x1, y1, x2, y2).
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Normalise to [0, 1] range for resolution independence.
            bbox = (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    severity=_get_severity(conf),
                    bbox=bbox,
                )
            )

    # Sort highest confidence first so the LLM agent sees the most certain
    # findings at the top of the detections JSON in its prompt.
    return sorted(detections, key=lambda d: d.confidence, reverse=True)
