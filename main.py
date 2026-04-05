"""
main.py — Local Entry Point
============================

Run the InspectAI app locally:

    python main.py

Or via Docker Compose:

    docker compose up

The app will be available at http://localhost:7860

Environment variables
---------------------
OLLAMA_HOST     Ollama server URL (default: http://localhost:11434).
                docker-compose.yml sets this to http://ollama:11434.
LLM_BACKEND     "ollama" (default) | "hf" | "none"
HF_TOKEN        HF token — required when LLM_BACKEND=hf
"""

from app.ui import build_app

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
