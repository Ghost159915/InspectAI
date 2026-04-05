# InspectAI Setup — Ubuntu Linux + AMD RX 7700 XT (ROCm)

**Expected setup time:** 20 minutes
**Training time:** 30–40 minutes for 100 epochs

---

## Step 1: Install AMD ROCm

ROCm is AMD's GPU compute platform (like CUDA for NVIDIA).

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROCm 5.7 (latest stable for RX 7700 XT)
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 focal main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install -y rocm-dkms rocm-libs rocm-hip-sdk

# Verify installation
rocm-smi
```

Should print GPU info: `GPU 0: AMD Radeon RX 7700 XT`

---

## Step 2: Install Python 3.11

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

---

## Step 3: Clone InspectAI

```bash
cd ~
git clone https://github.com/Ghost159915/InspectAI.git
cd InspectAI
```

---

## Step 4: Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

---

## Step 5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install base requirements
pip install -r requirements.txt

# Install PyTorch with ROCm support (critical!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

Takes 5–10 minutes. Be patient.

---

## Step 6: Verify GPU Support

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0))"
```

**Should print:**
```
CUDA available: True
Device name: AMD Radeon RX 7700 XT
```

If it says `CUDA available: False`, the ROCm installation failed. Check:

```bash
# GPU should be visible
rocm-smi

# PyTorch should find it
python -c "import torch; print(torch.version.cuda)"  # should print 12.1 (ROCm)
```

---

## Step 7: Copy MVTec Dataset

The dataset (~5 GB) is not in git. Copy from your Mac:

```bash
# On Mac:
# USB: Copy ~/Desktop/Projects/Projects/InspectAI/mvtec_anomaly_detection/ to USB
# Network: Enable file sharing, copy via Finder

# On Ubuntu, paste it at:
~/InspectAI/mvtec_anomaly_detection/
```

**Verify it's there:**

```bash
ls mvtec_anomaly_detection/ | head -15
```

Should list: `bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper`

---

## Step 8: Run Quick Test (5 epochs)

```bash
python scripts/train.py --epochs 5
```

**Expected output:**

```
[train] Device: cuda:0 (AMD Radeon RX 7700 XT)
[train] ── Stage 1: Build YOLO dataset ─────────────────────────────
[train]   Defect samples : 1016
[train]   Good samples   : 2850
[train] ── Stage 3: Train ──────────────────────────────────────────────
[train]   epochs=5  batch=8  imgsz=640  device=cuda:0
```

Watch for:
- ✅ `device=cuda:0` (GPU is active)
- ✅ Loss values dropping (learning happening)
- ✅ ~1–2 batches/second (GPU throughput)

**Expected runtime:** ~5 min build + ~10 min training.

---

## Step 9: Run Full Training

```bash
python scripts/train.py
```

**What to expect:**

- Build dataset: 5 min
- Training 100 epochs: 25–35 min (with early stopping ~20–30 epochs)
- **Total: 30–40 minutes**

**Monitor progress:**

Watch the terminal. Key metrics to track:

```
Epoch 10/100  box_loss=1.45  cls_loss=2.30  dfl_loss=1.20
Epoch 20/100  box_loss=0.95  cls_loss=1.10  dfl_loss=0.85
Epoch 30/100  box_loss=0.78  cls_loss=0.65  dfl_loss=0.72
```

**Losses should drop steadily.** If they plateau, that's early stopping (model converged).

---

## Step 10: Check Results

After training finishes:

```bash
# View metrics table
tail -20 runs/inspectai_v1/results.csv

# View plots
ls runs/inspectai_v1/*.png

# View trained weights
ls -lh models/inspectai_yolov8.pt
```

**Expected metrics:**

```
mAP@0.5      : 0.50–0.70 (good!)
Precision    : 0.70–0.85
Recall       : 0.65–0.80
Inference    : 4–6 ms/image
```

---

## Step 11: Test Gradio UI (Optional)

```bash
# Make sure Ollama is running (on Mac or elsewhere)
# Or comment out the agent/RAG calls in ui.py for testing

python main.py
```

Opens at `http://127.0.0.1:7860` — upload images, see defect detection + severity.

---

## Troubleshooting

### GPU not detected

```bash
# Check ROCm can see the GPU
rocm-smi

# Check PyTorch PyTorch setup
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If still false, reinstall PyTorch:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 --force-reinstall
```

### Out of memory (OOM) during training

```bash
# Reduce batch size
python scripts/train.py --batch 4
```

The RX 7700 XT has 12 GB VRAM (plenty), but OOM can happen if other processes are running.

### Training very slow (< 0.5 batches/sec)

GPU likely isn't being used. Check:

```bash
# While training, in another terminal:
watch -n 1 rocm-smi
```

Should show GPU utilization > 80%. If not:
- Close other apps (Firefox, etc.)
- Check `nvidia-smi` isn't trying to run (NVIDIA drivers conflict with ROCm)
- Reinstall PyTorch

### "No module named torch"

Venv not activated:

```bash
source .venv/bin/activate
```

Should see `(.venv)` in prompt.

---

## Performance

Expected on RX 7700 XT:

| Stage | Time |
|-------|------|
| Dataset build | 5 min |
| Training 100 epochs (actual: ~20–30 with early stop) | 25–35 min |
| Validation + benchmark | 2–3 min |
| **Total** | **30–40 min** |

Compare:
- MacBook Air M2 (MPS): 2.5+ hours
- Google Colab T4: 1–2 hours
- **Your PC (RX 7700 XT):** **30–40 min** ✅

---

## Next Steps

1. ✅ Complete training
2. Check `training_curves.png` — smooth curves?
3. Verify mAP@0.5 > 0.50
4. Test UI: `python main.py`
5. Deploy to Hugging Face Spaces (optional)
6. Start LangRobot (Project 2)

---

## Quick Commands (Copy & Paste)

**First time setup:**
```bash
cd ~/InspectAI
source .venv/bin/activate
python scripts/train.py --epochs 5  # test
```

**Full training:**
```bash
source .venv/bin/activate
python scripts/train.py
```

**After training:**
```bash
python main.py  # test UI
```

Good luck! 🚀
