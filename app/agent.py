"""
agent.py — LLM Report Generation Agent
========================================

Takes structured detections from detect.py, retrieves relevant context from
the RAG knowledge base, and produces a validated inspection report dict.

LLM backend selection (via LLM_BACKEND env var)
------------------------------------------------
  "ollama"  (default) — local Ollama server; auto-falls back to rule-based
                        if Ollama is unreachable (e.g. on HF Spaces)
  "hf"                — HF Serverless Inference API (set HF_TOKEN secret)
  "none"              — deterministic rule-based report, no LLM call

For Hugging Face Spaces deployment set:
  LLM_BACKEND = hf
  HF_TOKEN    = <your token>          (Space secret)
  HF_LLM_MODEL = meta-llama/Llama-3.2-3B-Instruct   (or any chat model)
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from langchain_core.prompts import PromptTemplate

from app.detect import Detection
from app.rag import retrieve_context

# ── Configuration ──────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = 0.1

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
# Qwen2.5-7B-Instruct is the default: it's not gated (no terms-of-service wall),
# reliably available on HF Serverless Inference, and produces clean JSON output.
# Override with the HF_LLM_MODEL Space variable if you prefer a different model.
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# ── Prompt template ────────────────────────────────────────────────────────────

REPORT_PROMPT = PromptTemplate(
    input_variables=["detections_json", "rag_context", "timestamp"],
    template="""You are an industrial inspection AI assistant. Analyse the
detected defects and produce a structured inspection report in JSON format.

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
  "summary": "<1-2 sentence plain-language summary>",
  "defects": [
    {{
      "label": "<defect type>",
      "severity": "low" | "medium" | "high",
      "confidence": <float 0.0-1.0>,
      "recommended_action": "<specific, actionable instruction>"
    }}
  ],
  "notes": "<any caveats — empty string if none>"
}}

Rules:
- overall_status = "FAIL"   if ANY defect has severity "high"
- overall_status = "REVIEW" if ANY defect has severity "medium" (and none "high")
- overall_status = "PASS"   if ALL defects are "low" OR defect_count is 0
- recommended_action MUST be specific — reference the standards context above
- Do not add keys beyond those in the schema

Timestamp: {timestamp}
""",
)

# ── Rule-based fallback ────────────────────────────────────────────────────────

_RULE_ACTIONS: dict[str, dict[str, str]] = {
    "scratch": {
        "high": "Isolate component immediately. Measure scratch depth with a profilometer. Reject if depth exceeds 0.1 mm per ISO 1302.",
        "medium": "Flag for supervisor review. Document scratch location and dimensions. Hold until engineering sign-off.",
        "low": "Log occurrence. Continue with increased monitoring frequency. Re-inspect at next scheduled interval.",
    },
    "pit": {
        "high": "Remove from production line. Pit likely exceeds void-size tolerance. Scrap or escalate for rework assessment.",
        "medium": "Measure pit diameter. Reject if larger than 0.5 mm per ISO 5436 surface-texture standards.",
        "low": "Document occurrence and continue. Monitor for cluster patterns indicating process drift.",
    },
    "crack": {
        "high": "Immediate quarantine. Structural crack detected — component must be scrapped. Investigate upstream process.",
        "medium": "Stop production on affected batch. Do not use part. Inspect tooling and forming parameters for root cause.",
        "low": "Flag for engineering review. Assess crack propagation risk before proceeding to next assembly stage.",
    },
    "contamination": {
        "high": "Stop line. Identify and remove contamination source. Clean and re-inspect entire affected batch before resuming.",
        "medium": "Clean surface with approved solvent per process spec. Re-inspect before continuing. Log batch ID.",
        "low": "Clean surface, document, and continue. Increase inspection frequency to detect repeat occurrences.",
    },
    "dent": {
        "high": "Component fails dimensional tolerance. Scrap. Check forming tooling for wear or misalignment.",
        "medium": "Measure deformation depth with CMM. Escalate to quality engineer if deformation exceeds 0.3 mm.",
        "low": "Document and monitor. Re-inspect before final assembly. Flag if count increases across consecutive parts.",
    },
}


def _rule_based_report(detections: list[Detection]) -> dict:
    """Generate a structured report without calling any LLM."""
    has_high = any(d.severity == "high" for d in detections)
    has_medium = any(d.severity == "medium" for d in detections)
    status = "FAIL" if has_high else ("REVIEW" if has_medium else "PASS")

    defect_entries = []
    for d in detections:
        action = _RULE_ACTIONS.get(d.label, {}).get(
            d.severity,
            "Inspect and document. Escalate to quality engineer if defect persists.",
        )
        defect_entries.append(
            {
                "label": d.label,
                "severity": d.severity,
                "confidence": round(d.confidence, 3),
                "recommended_action": action,
            }
        )

    count = len(detections)
    labels = ", ".join(sorted({d.label for d in detections}))
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": status,
        "defect_count": count,
        "summary": f"{count} defect{'s' if count != 1 else ''} detected ({labels}). Status: {status}.",
        "defects": defect_entries,
        "notes": "Report generated by rule-based system (LLM unavailable).",
    }


# ── HF Inference API ──────────────────────────────────────────────────────────


def _call_hf_llm(prompt_text: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model=HF_LLM_MODEL,
        token=os.getenv("HF_TOKEN"),
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=1024,
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content


# ── Ollama ─────────────────────────────────────────────────────────────────────

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import OllamaLLM

        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=LLM_TEMPERATURE,
        )
    return _llm


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return text


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_report(detections: list[Detection]) -> dict:
    """
    Generate a structured inspection report from detection results.

    Tries the configured LLM backend first; falls back to the rule-based
    report generator if the LLM is unreachable or returns invalid JSON.
    """
    if not detections:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "PASS",
            "defect_count": 0,
            "summary": "No defects detected. Surface appears nominal.",
            "defects": [],
            "notes": "",
        }

    if LLM_BACKEND == "none":
        return _rule_based_report(detections)

    # Build shared prompt inputs
    rag_query = "defect types: " + ", ".join(sorted({d.label for d in detections}))
    rag_context = retrieve_context(rag_query)
    detections_json = json.dumps([d.to_dict() for d in detections], indent=2)
    prompt_text = REPORT_PROMPT.format(
        detections_json=detections_json,
        rag_context=rag_context,
        timestamp=datetime.utcnow().isoformat(),
    )

    if LLM_BACKEND == "hf":
        try:
            return json.loads(_extract_json(_call_hf_llm(prompt_text)))
        except Exception as exc:
            error_msg = f"HF Inference API failed — {type(exc).__name__}: {exc}"
            print(f"[agent] {error_msg}")
            report = _rule_based_report(detections)
            report["notes"] = f"Rule-based fallback. Reason: {error_msg}"
            return report

    # Ollama (default)
    try:
        raw = _get_llm().invoke(prompt_text)
        return json.loads(_extract_json(raw))
    except Exception as exc:
        error_msg = f"Ollama unavailable — {type(exc).__name__}: {exc}"
        print(f"[agent] {error_msg}")
        report = _rule_based_report(detections)
        report["notes"] = f"Rule-based fallback. Reason: {error_msg}"
        return report
