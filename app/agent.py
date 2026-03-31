"""
agent.py — LLM Report Generation Agent
========================================

This module is the second stage of the InspectAI pipeline. It takes the
structured detections from ``detect.py``, retrieves relevant context from the
knowledge base via ``rag.py``, and uses a locally-running LLM (via Ollama) to
produce a validated inspection report as a Python dict.

Pipeline position:
    detect.py → [agent.py] ↔ rag.py → ui.py

Design decisions:
    Local LLM (Ollama):
        InspectAI deliberately avoids cloud LLM APIs. Using Ollama means no
        API keys, no cost per call, no data leaving the machine, and no
        internet dependency. The default model is Llama 3.2 (3B), which fits
        comfortably in ~4 GB of RAM and produces consistent structured output.
        Swap it for any Ollama-compatible model by changing ``OLLAMA_MODEL``.

    Structured output via prompt engineering:
        Rather than relying on function-calling or JSON-mode (which not all
        local models support reliably), the prompt specifies the exact JSON
        schema the model must follow, accompanied by strict rules. The
        ``_extract_json()`` helper strips any markdown fencing the model
        may add before parsing.

    Deterministic severity logic:
        The overall_status field (PASS / REVIEW / FAIL) is determined by
        strict rules in the prompt rather than left to the model's judgement.
        This makes the output predictable and auditable.

    RAG integration:
        Before calling the LLM, the agent queries the RAG knowledge base with
        a summary of the detected defect types. The returned context is
        injected into the prompt so the model can reference correct recommended
        actions from the defect standards document — rather than hallucinating
        generic advice.

Dependencies:
    langchain-ollama  — Ollama LLM integration for LangChain
    langchain-core    — PromptTemplate for structured prompt construction
    app.detect        — Detection dataclass (input type)
    app.rag           — retrieve_context (knowledge base lookup)
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from app.detect import Detection
from app.rag import retrieve_context

# ── Configuration ─────────────────────────────────────────────────────────────

# Ollama model tag. Must be pulled before first run: `ollama pull llama3.2`
# Alternatives: "mistral", "llama3.1:8b", "gemma2:2b"
OLLAMA_MODEL = "llama3.2"

# Ollama server address. Overridden at runtime by the OLLAMA_HOST env var,
# which docker-compose.yml sets to "http://ollama:11434" so the app container
# can reach the ollama service container by name.
OLLAMA_HOST = "http://localhost:11434"

# LLM temperature. 0.1 keeps output near-deterministic for structured JSON
# tasks — low enough to be consistent, not zero so the model doesn't get stuck.
LLM_TEMPERATURE = 0.1


# ── Prompt Template ───────────────────────────────────────────────────────────

# The prompt is split into clearly labelled sections:
#   1. Role definition   — sets the model's persona and task
#   2. Detected defects  — injected JSON from detect.py
#   3. RAG context       — retrieved knowledge base excerpts from rag.py
#   4. Output schema     — exact JSON structure the model must produce
#   5. Rules             — deterministic constraints for overall_status
#
# The schema is embedded as a literal dict with escaped braces ({{ }}) to
# avoid conflict with Python's str.format() syntax inside PromptTemplate.
REPORT_PROMPT = PromptTemplate(
    input_variables=["detections_json", "rag_context", "timestamp"],
    template="""You are an industrial inspection AI assistant. Your job is to
analyse the list of detected defects and produce a clear, structured
inspection report in JSON format.

## Detected Defects
{detections_json}

## Relevant Standards & Context
{rag_context}

## Instructions
Return ONLY valid JSON matching this schema exactly — no extra keys, no markdown:
{{
  "timestamp": "<ISO timestamp>",
  "overall_status": "PASS" | "REVIEW" | "FAIL",
  "defect_count": <integer>,
  "summary": "<1-2 sentence plain-language summary for a non-technical operator>",
  "defects": [
    {{
      "label": "<defect type>",
      "severity": "low" | "medium" | "high",
      "confidence": <float 0.0-1.0>,
      "recommended_action": "<specific, actionable instruction for the operator>"
    }}
  ],
  "notes": "<any additional observations or caveats — empty string if none>"
}}

Rules (apply these exactly — do not deviate):
- overall_status = "FAIL"   if ANY defect has severity "high"
- overall_status = "REVIEW" if ANY defect has severity "medium" (and none are "high")
- overall_status = "PASS"   if ALL defects are "low" OR defect_count is 0
- recommended_action MUST be specific — never write "inspect further" without detail
- Use the Relevant Standards & Context section above to inform recommended_action
- Do not add any keys beyond those in the schema

Timestamp: {timestamp}
""",
)


# ── LLM Loader ────────────────────────────────────────────────────────────────

# Module-level cache. Initialised on first call to _get_llm().
# This avoids reconstructing the LangChain LLM object on every request, which
# incurs a small but unnecessary overhead.
_llm: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    """
    Initialise and cache the Ollama LLM client.

    Reads ``OLLAMA_HOST`` from the environment on each call so that
    docker-compose's service-name routing (``http://ollama:11434``) is
    respected when running inside a container.

    Returns:
        Configured ``OllamaLLM`` instance ready for ``.invoke()`` calls.
    """
    global _llm
    if _llm is None:
        # Allow runtime override — critical for docker-compose networking.
        host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=host,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """
    Strip markdown code fences from an LLM response string.

    Many open-source LLMs wrap JSON output in markdown fences even when
    explicitly told not to::

        ```json
        { "key": "value" }
        ```

    This function removes the first and last lines if they are fence markers,
    returning the raw JSON string for ``json.loads()``.

    Args:
        text: Raw LLM response string, possibly wrapped in markdown fences.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove the opening fence line (e.g. "```json") and the closing "```"
        text = "\n".join(lines[1:-1])
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(detections: list[Detection]) -> dict:
    """
    Generate a structured inspection report from a list of defect detections.

    This is the primary entry point for the agent stage. It orchestrates:
        1. Early return for zero-detection "clean" images (no LLM call needed).
        2. RAG retrieval — queries the knowledge base with the detected labels
           to fetch relevant defect standard excerpts.
        3. Prompt construction — injects detections JSON and RAG context into
           ``REPORT_PROMPT``.
        4. LLM inference — sends the prompt to the local Ollama instance.
        5. Response parsing — strips markdown fencing and parses JSON.
        6. Validation — raises ``ValueError`` with the raw response if the
           model returned unparseable JSON, so the error surfaces cleanly in
           the UI rather than causing a cryptic downstream crash.

    Args:
        detections: List of ``Detection`` instances from ``detect.run_detection()``.
                    An empty list is valid and returns a PASS report immediately
                    without invoking the LLM.

    Returns:
        Parsed inspection report as a Python dict conforming to the schema
        defined in ``REPORT_PROMPT``. Guaranteed to have these keys:
        ``timestamp``, ``overall_status``, ``defect_count``, ``summary``,
        ``defects``, ``notes``.

    Raises:
        ValueError: If the LLM returns a response that cannot be parsed as
                    valid JSON. The error message includes the raw response for
                    debugging.
        ConnectionError: Propagated from Ollama if the LLM server is unreachable.
                         Ensure Ollama is running: ``ollama serve``

    Example::

        from app.detect import run_detection
        from app.agent import generate_report
        from PIL import Image

        img = Image.open("data/samples/scratch_example.jpg")
        detections = run_detection(img)
        report = generate_report(detections)
        print(report["overall_status"])  # "FAIL", "REVIEW", or "PASS"
    """
    # Fast path: no defects → return a clean PASS report without calling LLM.
    # This is both an optimisation and a UX improvement — clean images get an
    # instant response.
    if not detections:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "PASS",
            "defect_count": 0,
            "summary": "No defects detected. Surface appears nominal.",
            "defects": [],
            "notes": "",
        }

    # Build a natural-language RAG query from detected class names.
    # Deduplicating labels (set()) avoids redundant retrieval for repeated
    # detections of the same defect type (e.g. multiple scratches).
    rag_query = "defect types: " + ", ".join(sorted(set(d.label for d in detections)))
    rag_context = retrieve_context(rag_query)

    # Serialise detections to indented JSON for readability in the prompt.
    detections_json = json.dumps([d.to_dict() for d in detections], indent=2)
    timestamp = datetime.utcnow().isoformat()

    # Fill the prompt template with runtime values.
    prompt = REPORT_PROMPT.format(
        detections_json=detections_json,
        rag_context=rag_context,
        timestamp=timestamp,
    )

    llm = _get_llm()
    raw_response: str = llm.invoke(prompt)

    # Strip potential markdown fencing before JSON parsing.
    json_str = _extract_json(raw_response)

    try:
        report = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\n\nRaw LLM response:\n{raw_response}"
        ) from e

    return report
