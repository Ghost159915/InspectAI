"""
ui.py — Gradio Web Interface
==============================

Wires the detection and agent modules to a Gradio Blocks UI.
Annotates the uploaded image with colour-coded bounding boxes and
displays the structured inspection report alongside it.
"""

from __future__ import annotations

import json

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from app.detect import run_detection, Detection
from app.agent import generate_report

# ── Visual config ──────────────────────────────────────────────────────────────

SEVERITY_COLOURS: dict[str, str] = {
    "low":    "#22c55e",   # green-500
    "medium": "#f59e0b",   # amber-500
    "high":   "#ef4444",   # red-500
}

STATUS_EMOJI: dict[str, str] = {
    "PASS":   "✅ PASS",
    "REVIEW": "⚠️  REVIEW",
    "FAIL":   "❌ FAIL",
}

BBOX_LINE_WIDTH   = 3
LABEL_CHAR_WIDTH  = 7
LABEL_HEIGHT      = 20

# ── Annotation ─────────────────────────────────────────────────────────────────

def _annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    out  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        colour   = SEVERITY_COLOURS.get(det.severity, "#ffffff")

        draw.rectangle([px1, py1, px2, py2], outline=colour, width=BBOX_LINE_WIDTH)

        label_text = f"{det.label} ({det.severity}) {det.confidence:.0%}"
        bg_x2 = px1 + len(label_text) * LABEL_CHAR_WIDTH
        draw.rectangle([px1, py1 - LABEL_HEIGHT, bg_x2, py1], fill=colour)
        draw.text((px1 + 2, py1 - LABEL_HEIGHT + 2), label_text, fill="white")

    return out

# ── Pipeline ───────────────────────────────────────────────────────────────────

def inspect(image: np.ndarray) -> tuple[Image.Image, str]:
    """Full pipeline: detect → agent → annotate → format report."""
    pil_image  = Image.fromarray(image)
    detections = run_detection(pil_image)
    report     = generate_report(detections)
    annotated  = _annotate(pil_image, detections)

    status      = report.get("overall_status", "UNKNOWN")
    status_line = STATUS_EMOJI.get(status, status)
    report_md   = f"## {status_line}\n\n```json\n{json.dumps(report, indent=2)}\n```"

    return annotated, report_md

# ── App builder ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    import os
    from pathlib import Path

    with gr.Blocks(title="InspectAI", theme=gr.themes.Soft()) as app:

        gr.Markdown(
            """
            # 🔍 InspectAI — Visual Inspection System
            Upload a surface image to detect defects and generate an automated inspection report.
            *Powered by YOLOv8 + LLM agent + RAG — runs entirely on-device.*
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                run_btn = gr.Button("Run Inspection", variant="primary", size="lg")

            with gr.Column():
                output_image  = gr.Image(label="Annotated Result", type="pil")
                output_report = gr.Markdown(
                    label="Inspection Report",
                    value="*Upload an image and click Run Inspection to see results.*",
                )

        # Sample images strip — use any .jpg found in data/samples/
        sample_dir = Path("data/samples")
        samples = sorted(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []
        if samples:
            gr.Examples(
                examples=[[str(p)] for p in samples],
                inputs=input_image,
                label="Try a sample image",
            )

        run_btn.click(
            fn=inspect,
            inputs=input_image,
            outputs=[output_image, output_report],
        )

    return app
