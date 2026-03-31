# InspectAI 🔍

> **Agentic Visual Inspection System** — upload an image, get a structured defect report powered by computer vision and an LLM agent.

[![CI](https://github.com/YOUR_USERNAME/InspectAI/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/InspectAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- Add a demo GIF here once the app is running:
![InspectAI Demo](assets/demo.gif)
-->

---

## What It Does

InspectAI is a production-style AI pipeline that simulates an industrial visual inspection system. Given an image of a surface or component, it:

1. **Detects defects** using a fine-tuned YOLOv8 model trained on the MVTec Anomaly Detection dataset
2. **Localises and classifies** each defect (type, severity, bounding box)
3. **Generates a structured inspection report** via an LLM agent (runs locally via Ollama — no API keys needed)
4. **Cross-references defect standards** using a RAG knowledge base (FAISS + LangChain)
5. **Exposes a clean Gradio web UI** that any non-technical operator could use

**This runs entirely on your local machine.** No cloud services required.

---

## Architecture

```
Image Input
    │
    ▼
┌─────────────────────┐
│  YOLOv8 Detector    │  ← Fine-tuned on MVTec AD dataset
│  (detect.py)        │    Returns: [label, confidence, bbox]
└─────────┬───────────┘
          │ detections
          ▼
┌─────────────────────┐     ┌──────────────────────┐
│  LLM Agent          │────▶│  RAG Knowledge Base  │
│  (agent.py)         │     │  (rag.py)            │
│  Ollama / Llama 3   │◀────│  FAISS + LangChain   │
└─────────┬───────────┘     └──────────────────────┘
          │ structured report
          ▼
┌─────────────────────┐
│  Gradio Web UI      │  ← Runs at localhost:7860
│  (ui.py)            │
└─────────────────────┘
```

---

## Quick Start

### Option A — Docker (recommended, works on Mac and Windows)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/InspectAI.git
cd InspectAI

# 2. Run the full stack
docker compose up
```

Open [http://localhost:7860](http://localhost:7860) — that's it.

---

### Option B — Local Python

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/InspectAI.git
cd InspectAI

# 2. Create a virtual environment
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local LLM (one-time, ~4GB)
ollama pull llama3.2

# 5. Run the app
python main.py
```

Open [http://localhost:7860](http://localhost:7860).

---

## Project Structure

```
InspectAI/
├── app/
│   ├── detect.py          # YOLOv8 inference — defect detection
│   ├── agent.py           # LLM agent — report generation
│   ├── rag.py             # RAG pipeline — knowledge base retrieval
│   └── ui.py              # Gradio interface
├── data/
│   ├── samples/           # Sample test images
│   └── knowledge_base/    # Defect standard documents for RAG
├── models/                # Downloaded model weights (gitignored)
├── notebooks/
│   └── training.ipynb     # YOLOv8 fine-tuning walkthrough
├── tests/
│   ├── test_detect.py
│   ├── test_agent.py
│   └── test_rag.py
├── .github/workflows/
│   └── ci.yml             # Lint + unit tests on every push
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.py                # Entry point
```

---

## Dataset & Model

The detection model is fine-tuned on the **[MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)** — an industry-standard benchmark for surface defect detection covering 15 object/texture categories.

See [`notebooks/training.ipynb`](notebooks/training.ipynb) for the full fine-tuning walkthrough including:
- Dataset preparation and class mapping
- YOLOv8 training configuration
- Evaluation: precision, recall, mAP@0.5
- Export to ONNX for cross-platform inference

---

## Evaluation

| Metric | Value |
|---|---|
| mAP@0.5 | *TBD — fill after training* |
| Precision | *TBD* |
| Recall | *TBD* |
| RAG answer quality (Ragas) | *TBD* |
| Avg inference time (CPU) | *TBD* |

---

## Roadmap

- [x] Project scaffold and Docker setup
- [ ] YOLOv8 fine-tuning on MVTec AD
- [ ] LLM agent integration with Ollama
- [ ] RAG knowledge base with defect standards
- [ ] Gradio UI with image upload and annotated output
- [ ] Evaluation pipeline (Ragas for RAG, mAP for detection)
- [ ] Hugging Face Spaces deployment

---

## License

MIT — see [LICENSE](LICENSE).
