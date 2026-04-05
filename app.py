"""
app.py — Hugging Face Spaces Entry Point
=========================================

HF Spaces runs this file directly. It builds the Gradio app and launches it.
For local development use main.py instead (sets server_name / port explicitly).
"""

from app.ui import build_app

demo = build_app()
demo.launch()
