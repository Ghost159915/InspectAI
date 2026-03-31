"""
ui.py — Gradio Web Interface
==============================

This module is the final stage of the InspectAI pipeline. It wires the
detection and agent modules to a Gradio web UI, providing a browser-based
interface that any operator — technical or not — can use.

Pipeline position:
    detect.py → agent.py → rag.py → [ui.py]

Responsibilities:
    1. Accept an uploaded image from the user.
    2. Pass it through the full pipeline: detect → agent (which internally
       calls rag) → annotate → format.
    3. Return the annotated image (bounding boxes drawn) and the formatted
       inspection report side-by-side.

Gradio overview:
    Gradio provides an instant web UI from Python functions. ``gr.Blocks``
    gives full layout control (vs the simpler ``gr.Interface``). The app
    is served at ``localhost:7860`` and is accessible from any browser on
    the same machine. Setting ``server_name="0.0.0.0"`` in ``main.py``
    makes it reachable from outside the container when running via Docker.

Bounding box annotation:
    PIL's ``ImageDraw`` is used to draw colour-coded bounding boxes directly
    onto the image. Colours are mapped from severity:
        green  (#22c55e) → low
        amber  (#f59e0b) → medium
        red    (#ef4444) → high
    These are Tailwind CSS colours, chosen for accessibility contrast ratios.

    Bounding boxes stored in ``Detection.bbox`` are normalised [0, 1].
    This module rescales them to pixel coordinates using the image dimensions
    before drawing. This makes the annotation code resolution-independent.

Dependencies:
    gradio  — Web UI framework
    Pillow  — Image annotation (bounding boxes + labels)
    app.detect — run_detection (first pipeline stage)
    app.agent  — generate_report (second pipeline stage)
"""

from __future__ import annotations

import json

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from app.detect import run_detection, Detection
from app.agent import generate_report

# ── Visual Configuration ──────────────────────────────────────────────────────

# Bounding box outline colours mapped to severity level.
# Using Tailwind 500-weight colours for consistent, accessible contrast.
SEVERITY_COLOURS: dict[str, str] = {
    "low": "#22c55e",      # green-500
    "medium": "#f59e0b",   # amber-500
    "high": "#ef4444",     # red-500
}

# Overall status display strings shown as the report header in the UI.
STATUS_EMOJI: dict[str, str] = {
    "PASS":   "✅ PASS",
    "REVIEW": "⚠️  REVIEW",
    "FAIL":   "❌ FAIL",
}

# Thickness of bounding box outlines in pixels.
BBOX_LINE_WIDTH = 3

# Approximate character width in pixels for label background sizing.
# Used to draw a filled rectangle behind the label text for readability.
LABEL_CHAR_WIDTH_PX = 7
LABEL_HEIGHT_PX = 20


# ── Annotation ────────────────────────────────────────────────────────────────

def _annotate_image(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """
    Draw colour-coded bounding boxes and label overlays onto an image.

    Creates a copy of the input image so the original is never mutated.
    Converts to RGB to ensure consistent drawing regardless of the source
    image mode (e.g. RGBA PNGs will render correctly).

    For each detection:
        1. Scales the normalised bbox coordinates to pixel space.
        2. Draws a coloured rectangle outline with ``BBOX_LINE_WIDTH`` thickness.
        3. Draws a filled rectangle above the box as a label background.
        4. Draws the label text: "<class> (<severity>) <confidence%>".

    Args:
        image:      PIL Image to annotate. Not mutated.
        detections: List of ``Detection`` instances from the detection stage.
                    An empty list returns a copy of the original image unchanged.

    Returns:
        New PIL Image with all bounding boxes and labels drawn.
    """
    # Work on a copy to avoid mutating the input.
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for det in detections:
        # Scale normalised [0, 1] bbox coords → pixel coordinates.
        x1, y1, x2, y2 = det.bbox
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        colour = SEVERITY_COLOURS.get(det.severity, "#ffffff")

        # Bounding box outline.
        draw.rectangle([px1, py1, px2, py2], outline=colour, width=BBOX_LINE_WIDTH)

        # Label: "<class> (<severity>) <confidence%>"
        label_text = f"{det.label} ({det.severity}) {det.confidence:.0%}"

        # Filled background rectangle for the label text.
        # Placed immediately above the top edge of the bounding box.
        label_bg_x2 = px1 + len(label_text) * LABEL_CHAR_WIDTH_PX
        draw.rectangle(
            [px1, py1 - LABEL_HEIGHT_PX, label_bg_x2, py1],
            fill=colour,
        )

        # White label text on the coloured background.
        draw.text((px1 + 2, py1 - LABEL_HEIGHT_PX + 2), label_text, fill="white")

    return annotated


# ── Pipeline Orchestration ────────────────────────────────────────────────────

def inspect(image: np.ndarray) -> tuple[Image.Image, str]:
    """
    Run the full InspectAI pipeline on a single uploaded image.

    This function is the Gradio event handler — it is called automatically
    when the user clicks "Run Inspection". Gradio passes the uploaded image
    as a numpy array (H, W, 3) in RGB order.

    Processing steps:
        1. Convert numpy array → PIL Image (required by detect.py).
        2. ``run_detection()`` — YOLOv8 inference, returns list of Detection.
        3. ``generate_report()`` — LLM agent produces structured JSON report.
           Internally calls ``retrieve_context()`` from rag.py.
        4. ``_annotate_image()`` — draws bounding boxes on the PIL image.
        5. Format the report dict as a Markdown string for the UI.

    Args:
        image: numpy array from Gradio's image component, shape (H, W, 3),
               dtype uint8, values in [0, 255], RGB colour order.

    Returns:
        Tuple of:
            - annotated_image: PIL Image with bounding boxes drawn.
            - report_markdown: Formatted Markdown string displaying the status
              emoji header and the JSON report in a fenced code block.

    Note:
        Gradio renders the annotated PIL Image as the left panel output and
        the Markdown string in the right panel. Any exception raised here will
        be caught by Gradio and displayed as an error message in the UI.
    """
    # Gradio provides numpy arrays; detect.py accepts PIL Images.
    pil_image = Image.fromarray(image)

    # Stage 1: Defect detection.
    detections = run_detection(pil_image)

    # Stage 2: LLM report generation (includes RAG retrieval internally).
    report = generate_report(detections)

    # Stage 3: Annotate the image with detection results.
    annotated = _annotate_image(pil_image, detections)

    # Stage 4: Format report for Markdown display.
    status = report.get("overall_status", "UNKNOWN")
    status_line = STATUS_EMOJI.get(status, status)
    # Embed the full JSON in a fenced code block so Gradio renders it with
    # syntax highlighting.
    report_markdown = (
        f"## {status_line}\n\n"
        f"```json\n{json.dumps(report, indent=2)}\n```"
    )

    return annotated, report_markdown


# ── App Builder ───────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """
    Construct and return the Gradio application.

    Uses ``gr.Blocks`` for full layout control. The app is not launched here;
    ``main.py`` calls ``app.launch()`` with the appropriate server settings.

    Layout:
        - Header: Title and subtitle.
        - Two-column row:
            Left:  Image upload component + "Run Inspection" button.
            Right: Annotated result image + Markdown report panel.
        - Sample images strip: pre-loaded examples for quick demoing.

    The ``run_btn.click()`` wiring connects the "Run Inspection" button
    to the ``inspect()`` function, mapping input → outputs automatically.

    Returns:
        Configured ``gr.Blocks`` instance, ready for ``.launch()``.
    """
    with gr.Blocks(title="InspectAI", theme=gr.themes.Soft()) as app:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🔍 InspectAI — Visual Inspection System
            Upload a surface image to detect defects and generate an automated inspection report.
            *Powered by YOLOv8 + Llama 3 + RAG — runs entirely on your machine.*
            """
        )

        # ── Main Panel ────────────────────────────────────────────────────────
        with gr.Row():
            # Left column: input controls
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",   # Gradio converts to numpy before calling inspect()
                    sources=["upload", "clipboard"],  # Allow paste from clipboard
                )
                run_btn = gr.Button("Run Inspection", variant="primary", size="lg")

            # Right column: outputs
            with gr.Column():
                output_image = gr.Image(
                    label="Annotated Result",
                    type="pil",     # We return a PIL Image from inspect()
                )
                output_report = gr.Markdown(
                    label="Inspection Report",
                    value="*Upload an image and click Run Inspection to see results.*",
                )

        # ── Sample Images ──────────────────────────────────────────────────────
        # Pre-loaded examples let reviewers and recruiters demo the app without
        # needing to source their own images.
        # Only rendered if at least one sample file exists on disk.
        import os
        sample_path = "data/samples/scratch_example.jpg"
        if os.path.exists(sample_path):
            gr.Examples(
                examples=[sample_path],
                inputs=input_image,
                label="Try a sample image",
            )

        # ── Event Wiring ───────────────────────────────────────────────────────
        # Clicking the button calls inspect() with the image as input and
        # pipes the two return values to the two output components.
        run_btn.click(
            fn=inspect,
            inputs=input_image,
            outputs=[output_image, output_report],
        )

    return app
