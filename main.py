"""
main.py — InspectAI Entry Point
=================================

Start the InspectAI application by running::

    python main.py

Or via Docker Compose (recommended for cross-platform use)::

    docker compose up

The application will be available at http://localhost:7860

Environment variables:
    OLLAMA_HOST     Ollama server URL. Defaults to "http://localhost:11434".
                    docker-compose.yml sets this to "http://ollama:11434" so
                    the app container can reach the ollama service by name.
    GRADIO_SERVER_NAME
                    Host to bind Gradio's HTTP server. Set to "0.0.0.0" by
                    docker-compose.yml and the Dockerfile so the UI is
                    reachable from outside the container.

Notes:
    - ``share=False`` keeps the app local. Set to ``True`` to generate a
      temporary public Gradio URL (useful for sharing a demo without deploying
      to Hugging Face Spaces).
    - ``show_error=True`` surfaces Python tracebacks in the Gradio UI so
      errors are visible without needing to check server logs.
"""

from app.ui import build_app

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",   # Bind to all interfaces — required inside Docker
        server_port=7860,         # Default Gradio port
        share=False,              # Set True for a temporary public Gradio link
        show_error=True,          # Show Python errors in the browser UI
    )
