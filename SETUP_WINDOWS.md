# InspectAI Setup Guide — Windows PC with AMD RX 7700 XT

This guide walks you through setting up InspectAI on Windows to train on your AMD Ryzen 5600X + RX 7700 XT.

**Expected setup time:** ~30 minutes
**Training time:** ~20–40 minutes for 100 epochs (vs. 2.5+ hours on MacBook Air)

---

## Prerequisites

- Windows 10/11 (64-bit)
- AMD RX 7700 XT GPU (12 GB VRAM)
- At least 50 GB free disk space (for MVTec dataset)
- Stable internet connection

---

## Step 1: Install Python 3.11

**Option A: Direct download (recommended)**

1. Visit https://www.python.org/downloads/
2. Download **Python 3.11** (latest 3.11.x version)
3. Run the installer
4. **IMPORTANT:** Check the box `Add Python to PATH`
5. Click `Install Now`
6. Wait for installation to complete

**Verify installation:**

Open Command Prompt (Win+R, type `cmd`, press Enter) and run:

```cmd
python --version
```

Should print: `Python 3.11.x`

---

## Step 2: Clone the InspectAI Repository

1. Open Command Prompt or PowerShell
2. Navigate to where you want the project (e.g., `C:\Users\YourName\Desktop`):

```cmd
cd Desktop
```

3. Clone the repository:

```cmd
git clone https://github.com/Ghost159915/InspectAI.git
cd InspectAI
```

4. Verify you're in the right place:

```cmd
dir
```

Should show folders like `app/`, `scripts/`, `data/`, etc.

---

## Step 3: Create a Python Virtual Environment

A virtual environment isolates InspectAI's dependencies from your system Python.

```cmd
python -m venv .venv
```

This creates a `.venv` folder. Activating it:

```cmd
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of your command prompt.

---

## Step 4: Install Dependencies

With the venv activated, install the base requirements:

```cmd
pip install -r requirements.txt
```

This takes 2–5 minutes. You'll see a lot of output. Wait for it to finish.

**Verify installation:**

```cmd
python -c "import torch; print(torch.__version__)"
```

Should print something like `2.x.x`

---

## Step 5: Install AMD GPU Support (torch-directml)

This enables your RX 7700 XT to accelerate training on Windows.

```cmd
pip install torch-directml
```

Takes ~1 minute.

**Verify GPU support:**

```cmd
python -c "import torch_directml; print(torch_directml.device())"
```

Should print: `privateuseone:0` (or similar)

---

## Step 6: Transfer the MVTec Dataset

The MVTec AD dataset is ~5 GB and cannot be cloned from git. You need to copy it from your Mac.

**On your Mac:**

1. Open Finder
2. Navigate to: `~/Desktop/Projects/Projects/InspectAI/mvtec_anomaly_detection/`
3. Copy this folder

**Transfer options:**

- **USB Drive (fastest):** Copy to USB, physically carry to PC, paste into `C:\Users\YourName\Desktop\InspectAI\`
- **Network (if on same WiFi):** Enable file sharing on Mac, access via `\\Mac-IP\` in Windows Explorer
- **Cloud (Google Drive/Dropbox):** Drag folder to cloud storage on Mac, download on PC

**On your Windows PC:**

Paste the `mvtec_anomaly_detection/` folder so the structure looks like:

```
C:\Users\YourName\Desktop\InspectAI\
├── mvtec_anomaly_detection\       ← pasted here
├── app\
├── scripts\
├── data\
├── ...
```

**Verify dataset is present:**

```cmd
dir mvtec_anomaly_detection
```

Should list 15 folders: `bottle`, `cable`, `capsule`, etc.

---

## Step 7: Verify Everything Works

Before starting a full training run, do a quick sanity check:

```cmd
python scripts/train.py --epochs 5
```

This will:
1. ✅ Build the YOLO dataset from MVTec (5 min)
2. ✅ Train for 5 epochs (quick test)
3. ✅ Verify GPU acceleration is working
4. ✅ Print device info and metrics

**Expected output (look for these):**

```
[train] Device: privateuseone:0  (GPU is detected!)
[train] ── Stage 1: Build YOLO dataset ─────────────────────────────
[train]   Defect samples : 1016
[train]   Good samples   : 2850
[train] ── Stage 3: Train ──────────────────────────────────────────────
[train]   epochs=5  batch=8  imgsz=640  device=privateuseone:0
```

If you see `device=privateuseone:0`, the GPU is working! 🎉

---

## Step 8: Run Full Training

Once the sanity check passes, run the full 100-epoch training:

```cmd
python scripts/train.py
```

**What to expect:**

- **Build dataset:** ~5 min
- **Training:** ~25–40 min (depending on GPU utilization)
- **Total:** ~30–45 min

**Monitor progress:**

Watch for:
- ✅ `box_loss`, `cls_loss`, `dfl_loss` dropping (losses decreasing = learning)
- ✅ `mAP@0.5` rising during validation (accuracy improving)
- ✅ Early stopping at ~20–30 epochs (if model converges quickly)

**Common issues:**

| Issue | Solution |
|-------|----------|
| `RuntimeError: CUDA out of memory` | Reduce batch size: `python scripts/train.py --batch 4` |
| `torch_directml not found` | Re-run: `pip install torch-directml` |
| Dataset not found | Make sure `mvtec_anomaly_detection/` is in the InspectAI root folder |

---

## Step 9: Check Results

After training finishes, results are in:

```
runs/inspectai_v1/
├── weights/best.pt           ← best model (exported to models/)
├── training_curves.png        ← loss & mAP plots
├── results.csv                ← metrics per epoch
└── ...
```

The trained weights are automatically copied to `models/inspectai_yolov8.pt`.

**Expected metrics:**

| Metric | Target |
|--------|--------|
| mAP@0.5 | > 0.50 (good) |
| Precision | > 0.70 |
| Recall | > 0.65 |

---

## Step 10: Test the Web UI (Optional)

Once you have trained weights, you can test the Gradio UI locally:

```cmd
python main.py
```

Opens at `http://127.0.0.1:7860` — you can upload images and see defect detections.

---

## Troubleshooting

**Q: My GPU isn't being used (still says `device=cpu`)**

```cmd
python -c "import torch_directml; print('DirectML available:', torch_directml.is_available())"
```

If it returns `False`, reinstall:

```cmd
pip uninstall torch torch-directml
pip install torch torch-directml
```

**Q: Training is very slow (< 1 epoch per minute)**

Check GPU usage in Task Manager → Performance → GPU. If it's low (<20%), the GPU isn't being utilized. This can happen if:
- DirectML isn't installed (see above)
- Batch size is too small (try `--batch 16`)
- GPU drivers are outdated (update AMD drivers)

**Q: `git clone` doesn't work**

You need Git for Windows:
- Download from https://git-scm.com/download/win
- Install with defaults
- Restart Command Prompt

**Q: Out of memory errors during dataset build**

The dataset build is CPU-bound and shouldn't use much RAM. If you still hit OOM:

```cmd
python scripts/train.py --skip-build
```

(if you built it once already)

---

## Performance Comparison

| Device | 100 epochs |
|--------|-----------|
| MacBook Air M2 (MPS) | ~2.5 hours |
| Ryzen 5600X (CPU only) | ~6+ hours |
| **RX 7700 XT (DirectML)** | **~30–40 min** ✅ |

---

## Next Steps

1. ✅ Train the model
2. Test with `python main.py`
3. Deploy to Hugging Face Spaces (public demo)
4. Start LangRobot project (Project 2)

---

## Still stuck?

If you hit issues:
1. Check the error message carefully
2. Google the error (most are common)
3. Post the error + output to the repo issues
4. Or ask me directly with the full error output

Good luck! 🚀
