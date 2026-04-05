"""
scripts/train.py — InspectAI YOLOv8 Training Pipeline
======================================================

Builds a YOLO-format dataset from the local MVTec AD dataset, fine-tunes
YOLOv8n, evaluates metrics, and exports weights to models/inspectai_yolov8.pt.

Usage (run from the InspectAI project root)
-------------------------------------------
    # Full pipeline
    python scripts/train.py

    # Skip dataset build if yolo_dataset/ already exists
    python scripts/train.py --skip-build

    # Custom config
    python scripts/train.py --epochs 30 --batch 16

    # Quick sanity check
    python scripts/train.py --epochs 5

    # CPU override (auto-detected by default)
    python scripts/train.py --device cpu

Dataset location
----------------
    InspectAI/mvtec_anomaly_detection/   ← full 15-category MVTec AD dataset
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from PIL import Image
from tqdm import tqdm

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent          # InspectAI/
DATA_RAW  = ROOT / "mvtec_anomaly_detection"                # full MVTec dataset
DATA_YOLO = ROOT / "data" / "yolo_dataset"                  # processed dataset
MODEL_DIR = ROOT / "models"
RUNS_DIR  = ROOT / "runs"

# ── Class definitions (sync with app/detect.py + defect_standards.md) ─────────
CLASS_NAMES = ["scratch", "pit", "crack", "contamination", "dent"]
CLASS_ID    = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Full MVTec AD → InspectAI class mapping ────────────────────────────────────
# Covers all 15 categories. Defect types that don't map cleanly to our 5
# industrial classes (e.g. orientation errors, wrong pill type) are omitted.
#
# Format: { category: { mvtec_defect_folder: inspectai_class_name } }
# 'good' images are added automatically as negatives (empty label files).
MVTEC_MAP: dict[str, dict[str, str]] = {
    "bottle": {
        "contamination": "contamination",
        "broken_large":  "dent",         # large breakage = structural deformation
        "broken_small":  "pit",          # small chip = pit / void
    },
    "cable": {
        "bent_wire":           "dent",    # bent wire = deformation
        "cut_outer_insulation": "scratch", # surface cut = scratch
        "poke_insulation":     "pit",     # poke = small pit / puncture
    },
    "capsule": {
        "crack":   "crack",
        "poke":    "pit",                 # pinhole poke = pit
        "scratch": "scratch",
        "squeeze": "dent",               # squeeze deformation = dent
    },
    "carpet": {
        "cut":               "scratch",   # linear cut = scratch
        "hole":              "pit",
        "metal_contamination": "contamination",
    },
    "grid": {
        "bent":              "dent",
        "broken":            "dent",      # structural break = dent/deformation
        "metal_contamination": "contamination",
    },
    "hazelnut": {
        "crack": "crack",
        "cut":   "scratch",              # linear cut = scratch
        "hole":  "pit",
    },
    "leather": {
        "cut":  "scratch",               # linear cut = scratch
        "glue": "contamination",         # glue residue = contamination
        "poke": "pit",                   # pinhole = pit
    },
    "metal_nut": {
        "bent":    "dent",
        "color":   "contamination",      # discolouration = contamination
        "scratch": "scratch",
    },
    "pill": {
        "contamination": "contamination",
        "crack":         "crack",
        "scratch":       "scratch",
    },
    "screw": {
        "scratch_head": "scratch",
        "scratch_neck": "scratch",
    },
    "tile": {
        "crack":      "crack",
        "glue_strip": "contamination",   # glue residue = contamination
        "gray_stroke": "contamination",  # paint/coating mark = contamination
        "oil":        "contamination",
        "rough":      "dent",            # rough surface = deformation
    },
    "toothbrush": {
        "defective": "scratch",          # toothbrush defects are primarily bristle damage
    },
    "transistor": {
        "bent_lead":    "dent",          # bent lead = deformation
        "damaged_case": "dent",          # case damage = deformation
        "cut_lead":     "scratch",       # cut = scratch
    },
    "wood": {
        "hole":    "pit",
        "liquid":  "contamination",      # liquid stain = contamination
        "scratch": "scratch",
    },
    "zipper": {
        "squeezed_teeth": "dent",        # squeezed = deformation
        "split_teeth":    "dent",        # split = structural deformation
        "broken_teeth":   "scratch",     # broken teeth edge = scratch-like damage
    },
}

TRAIN_SPLIT = 0.80   # fraction of defect images → train


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def mask_to_yolo(mask_path: Path, class_id: int,
                 img_w: int, img_h: int) -> str | None:
    """
    Convert a binary segmentation mask to a YOLO bounding-box annotation.

    Finds the tight bounding rectangle of all non-zero pixels and returns a
    YOLO annotation line:  ``class_id  cx  cy  w  h``  (normalised [0, 1]).
    Returns None if the mask is missing or entirely black.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1)       / img_w
    h  = (y2 - y1)       / img_h

    # Clamp — masks occasionally extend 1–2 px beyond image bounds
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    w,  h  = max(0.001, min(1.0, w)), max(0.001, min(1.0, h))

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ── Stage 1: Build YOLO dataset ───────────────────────────────────────────────

def build_dataset() -> Path:
    """
    Walk the full MVTec AD dataset and build the YOLO folder layout:

        data/yolo_dataset/
        ├── images/{train,val}/*.jpg
        ├── labels/{train,val}/*.txt   ← empty file = no defect (negative)
        └── dataset.yaml

    Split: 80 % of defect images → train, 20 % → val.
            50 % of good (clean) images → each split.
    """
    _log("── Stage 1: Build YOLO dataset ─────────────────────────────────")

    if not DATA_RAW.exists():
        _log(f"  ✗ Dataset not found at {DATA_RAW}")
        _log("    Place the mvtec_anomaly_detection/ folder in the project root.")
        sys.exit(1)

    defect_samples: list[dict] = []
    good_samples:   list[dict] = []

    for cat, defect_map in MVTEC_MAP.items():
        cat_dir = DATA_RAW / cat
        if not cat_dir.exists():
            _log(f"  ⚠ {cat} not found — skipping")
            continue

        # ── Defect images ────────────────────────────────────────────────
        for defect_type, class_name in defect_map.items():
            class_id = CLASS_ID[class_name]
            img_dir  = cat_dir / "test" / defect_type
            mask_dir = cat_dir / "ground_truth" / defect_type

            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                # Ground-truth mask: same stem, suffix _mask.png
                mask_path = mask_dir / f"{img_path.stem}_mask.png"

                raw = cv2.imread(str(img_path))
                if raw is None:
                    continue
                img_h, img_w = raw.shape[:2]

                label_line = None
                if mask_path.exists():
                    label_line = mask_to_yolo(mask_path, class_id, img_w, img_h)

                # Fallback: no mask → centred 90% bbox. Model still learns
                # the class; spatial accuracy is lower but nothing is lost.
                if label_line is None:
                    label_line = f"{class_id} 0.500000 0.500000 0.900000 0.900000"

                defect_samples.append({
                    "img_path":    img_path,
                    "label_lines": [label_line],
                    "category":    cat,
                    "class_name":  class_name,
                })

        # ── Good (negative) images ───────────────────────────────────────
        for split_name in ("train", "test"):
            good_dir = cat_dir / split_name / "good"
            if not good_dir.exists():
                continue
            for img_path in sorted(good_dir.glob("*.png")):
                good_samples.append({
                    "img_path":    img_path,
                    "label_lines": [],   # empty = negative example
                    "category":    cat,
                    "class_name":  "good",
                })

    _log(f"  Defect samples : {len(defect_samples)}")
    _log(f"  Good samples   : {len(good_samples)}")

    if not defect_samples:
        _log("  ✗ No defect images found. Check MVTEC_MAP and dataset structure.")
        sys.exit(1)

    # ── Class balance report ──────────────────────────────────────────────
    class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for s in defect_samples:
        class_counts[s["class_name"]] += 1
    _log("  Class distribution (before oversampling):")
    for c, n in class_counts.items():
        _log(f"    {c:15s}: {n}")

    # ── Oversample minority classes ───────────────────────────────────────
    # Duplicate underrepresented class images until all classes have the
    # same count as the majority class. This prevents the model learning
    # "when in doubt, predict scratch" due to scratch having 3× more
    # examples than crack.
    by_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    for s in defect_samples:
        by_class[s["class_name"]].append(s)

    target_count = max(len(v) for v in by_class.values())
    _log(f"  Oversampling minority classes to {target_count} each …")

    balanced: list[dict] = []
    for class_name, samples in by_class.items():
        balanced.extend(samples)
        n_needed = target_count - len(samples)
        if n_needed > 0:
            pool   = samples * ((n_needed // len(samples)) + 1)
            copies = pool[:n_needed]
            for i, s in enumerate(copies):
                balanced.append({**s, "copy_idx": i + 1})
            _log(f"    {class_name:15s}: {len(samples):3d} → {target_count} (+{n_needed})")
        else:
            _log(f"    {class_name:15s}: {len(samples):3d} (no change)")

    defect_samples = balanced
    _log(f"  Total defect samples after balancing: {len(defect_samples)}")

    # ── Assign splits ─────────────────────────────────────────────────────
    random.shuffle(defect_samples)
    random.shuffle(good_samples)

    n_train_defect = int(len(defect_samples) * TRAIN_SPLIT)
    n_train_good   = len(good_samples) // 2

    for i, s in enumerate(defect_samples):
        s["split"] = "train" if i < n_train_defect else "val"
    for i, s in enumerate(good_samples):
        s["split"] = "train" if i < n_train_good else "val"

    all_samples = defect_samples + good_samples
    split_counts = {"train": 0, "val": 0}
    for s in all_samples:
        split_counts[s["split"]] += 1
    _log(f"  Split → train: {split_counts['train']}  val: {split_counts['val']}")

    # ── Write folder structure ────────────────────────────────────────────
    if DATA_YOLO.exists():
        shutil.rmtree(DATA_YOLO)

    for split in ("train", "val"):
        (DATA_YOLO / "images" / split).mkdir(parents=True)
        (DATA_YOLO / "labels" / split).mkdir(parents=True)

    errors = 0
    for sample in tqdm(all_samples, desc="  Writing dataset", unit="img"):
        split    = sample["split"]
        src      = sample["img_path"]
        stem     = f"{sample['category']}_{sample['class_name']}_{src.stem}"
        # Oversampled copies get a unique suffix so they don't overwrite originals
        copy_idx = sample.get("copy_idx")
        if copy_idx is not None:
            stem = f"{stem}_copy{copy_idx}"
        dst_img = DATA_YOLO / "images" / split / f"{stem}.jpg"
        dst_lbl = DATA_YOLO / "labels" / split / f"{stem}.txt"

        try:
            Image.open(src).convert("RGB").save(dst_img, "JPEG", quality=92)
        except Exception:
            errors += 1
            continue

        dst_lbl.write_text("\n".join(sample["label_lines"]))

    if errors:
        _log(f"  ⚠ {errors} images skipped (read errors)")

    # ── Write dataset.yaml ────────────────────────────────────────────────
    yaml_path = DATA_YOLO / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path":  str(DATA_YOLO),
                "train": "images/train",
                "val":   "images/val",
                "nc":    len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            f, default_flow_style=False, sort_keys=False,
        )

    for split in ("train", "val"):
        n = len(list((DATA_YOLO / "images" / split).glob("*.jpg")))
        _log(f"  {split}: {n} images written")

    _log(f"  ✓ dataset.yaml → {yaml_path}\n")
    return yaml_path


# ── Stage 2: Visualise ────────────────────────────────────────────────────────

def visualise_dataset() -> None:
    """Save a class-distribution bar chart to data/yolo_dataset/."""
    _log("── Stage 2: Visualise ──────────────────────────────────────────")

    counts = {"train": {c: 0 for c in CLASS_NAMES},
              "val":   {c: 0 for c in CLASS_NAMES}}
    for split in ("train", "val"):
        for lbl in (DATA_YOLO / "labels" / split).glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                if line:
                    counts[split][CLASS_NAMES[int(line.split()[0])]] += 1

    x     = np.arange(len(CLASS_NAMES))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_t = ax.bar(x - bar_w / 2,
                    [counts["train"][c] for c in CLASS_NAMES],
                    bar_w, label="train", color="steelblue", alpha=0.85)
    bars_v = ax.bar(x + bar_w / 2,
                    [counts["val"][c] for c in CLASS_NAMES],
                    bar_w, label="val", color="salmon", alpha=0.85)
    ax.bar_label(bars_t, padding=3, fontsize=9)
    ax.bar_label(bars_v, padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("Instances")
    ax.set_title("Class distribution — InspectAI YOLO dataset",
                 fontsize=13, fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    out = DATA_YOLO / "class_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    _log(f"  ✓ Chart saved → {out}\n")


# ── Stage 3: Train ────────────────────────────────────────────────────────────

def detect_device(override: str | None) -> str | int:
    """Return the best available compute device."""
    if override is not None:
        return int(override) if override.isdigit() else override

    import torch
    if torch.backends.mps.is_available():
        _log("  Device: Apple MPS  (~25–40 min / 50 epochs on M-series Mac)")
        return "mps"
    if torch.cuda.is_available():
        _log(f"  Device: CUDA — {torch.cuda.get_device_name(0)}")
        return 0
    _log("  Device: CPU  (slow — consider Google Colab for a free T4 GPU)")
    return "cpu"


def _make_min_delta_stopper(min_delta: float, patience: int):
    """
    Return an Ultralytics callback that stops training when mAP@0.5 fails to
    improve by *min_delta* within a rolling window of *patience* epochs.

    How it works
    ------------
    A reference value is set at the start of each window.  After every
    validation epoch we check whether the best mAP@0.5 seen since the window
    opened has risen by at least *min_delta*.  If yes, the window resets and
    training continues.  If *patience* epochs pass without that threshold being
    reached, ``trainer.stop = True`` is set, which Ultralytics checks at the
    end of each epoch and exits cleanly.

    Example: min_delta=0.1, patience=30, starting mAP@0.5=0.812
        → training stops unless mAP@0.5 reaches 0.912 within 30 epochs.
    """
    state: dict = {"reference": None, "best": -1.0, "no_improve_count": 0}

    def on_fit_epoch_end(trainer) -> None:
        current = float(trainer.metrics.get("metrics/mAP50(B)", 0.0))

        if state["reference"] is None:          # first epoch — initialise window
            state["reference"] = current
            state["best"]      = current
            return

        if current > state["best"]:
            state["best"] = current

        improvement = state["best"] - state["reference"]

        if improvement >= min_delta:
            # Threshold met — open a new window from the new high
            _log(f"  [stopper] mAP@0.5 improved +{improvement:.4f} ≥ {min_delta} "
                 f"— resetting patience window (best={state['best']:.4f})")
            state["reference"]      = state["best"]
            state["no_improve_count"] = 0
        else:
            state["no_improve_count"] += 1
            remaining = patience - state["no_improve_count"]
            _log(f"  [stopper] mAP@0.5 improvement {improvement:.4f} < {min_delta} "
                 f"— {state['no_improve_count']}/{patience} epochs "
                 f"({remaining} remaining before stop)")

            if state["no_improve_count"] >= patience:
                _log(f"  [stopper] Early stop triggered: mAP@0.5 did not improve "
                     f"by {min_delta} in {patience} epochs "
                     f"(best={state['best']:.4f}, ref={state['reference']:.4f})")
                trainer.stop = True

    return on_fit_epoch_end


def train(yaml_path: Path, epochs: int, batch: int,
          device: str | int, run_name: str,
          finetune_weights: Path | None = None,
          min_delta: float | None = None,
          min_delta_patience: int = 30) -> Path:
    """
    Train or fine-tune YOLOv8s. Returns path to best.pt.

    Parameters
    ----------
    finetune_weights : Path or None
        If given, load these weights instead of the base yolov8s.pt COCO
        checkpoint. Useful for continuing a previous training run.
    min_delta : float or None
        Minimum mAP@0.5 improvement required within *min_delta_patience*
        epochs before training is stopped early. None disables the custom
        stopper and falls back to Ultralytics' built-in patience.
    min_delta_patience : int
        Rolling window size (epochs) for the min-delta check.
    """
    from ultralytics import YOLO

    _log("── Stage 3: Train ──────────────────────────────────────────────")

    if finetune_weights is not None:
        if not finetune_weights.exists():
            _log(f"  ✗ Finetune weights not found: {finetune_weights}")
            sys.exit(1)
        _log(f"  Mode   : fine-tune from {finetune_weights}")
        model = YOLO(str(finetune_weights))
    else:
        _log("  Mode   : train from scratch (yolov8s COCO weights)")
        model = YOLO("yolov8s.pt")

    _log(f"  epochs={epochs}  batch={batch}  imgsz=640  device={device}")
    if min_delta is not None:
        _log(f"  Early stop: mAP@0.5 must improve by ≥{min_delta} "
             f"within {min_delta_patience} epochs")
        model.add_callback("on_fit_epoch_end",
                           _make_min_delta_stopper(min_delta, min_delta_patience))
        # Disable the built-in patience stopper so only our callback fires.
        builtin_patience = epochs + 1
    else:
        builtin_patience = max(10, epochs // 5)

    model.train(
        data      = str(yaml_path),
        epochs    = epochs,
        imgsz     = 640,
        batch     = batch,
        patience  = builtin_patience,
        device    = device,
        project   = str(RUNS_DIR),
        name      = run_name,
        exist_ok  = True,
        save      = True,
        plots     = True,
        verbose   = True,
        # Augmentation
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        mosaic    = 1.0,
        cls       = 0.5,
    )

    best = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best.exists():
        _log(f"  ✗ best.pt not found at {best}")
        sys.exit(1)

    _log(f"  ✓ Training complete. Best weights: {best}\n")
    return best


# ── Stage 4: Evaluate ─────────────────────────────────────────────────────────

def evaluate(best_weights: Path, yaml_path: Path,
             device: str | int) -> dict[str, float]:
    """Run validation metrics + inference latency benchmark."""
    from ultralytics import YOLO

    _log("── Stage 4: Evaluate ───────────────────────────────────────────")

    model   = YOLO(str(best_weights))
    metrics = model.val(data=str(yaml_path), split="val",
                        device=device, verbose=False)

    map50   = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    prec    = float(metrics.box.mp)
    recall  = float(metrics.box.mr)

    _log(f"  mAP@0.5      : {map50:.3f}")
    _log(f"  mAP@0.5:0.95 : {map5095:.3f}")
    _log(f"  Precision    : {prec:.3f}")
    _log(f"  Recall       : {recall:.3f}")

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        _log("  Per-class AP@0.5:")
        for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            _log(f"    {CLASS_NAMES[i]:15s}: {ap:.3f}")

    # Inference latency
    # Load a fresh model instance — model.val() leaves internal state that
    # breaks subsequent inference on MPS (PyTorch bug: "tensors do not track
    # version counter"). A fresh load sidesteps this completely.
    val_imgs = sorted((DATA_YOLO / "images" / "val").glob("*.jpg"))
    elapsed_ms = 0.0
    if val_imgs:
        bench_model = YOLO(str(best_weights))
        img = Image.open(val_imgs[0])
        for _ in range(3):                      # warm-up
            bench_model(img, verbose=False)
        N     = 20
        start = time.perf_counter()
        for _ in range(N):
            bench_model(img, verbose=False)
        elapsed_ms = (time.perf_counter() - start) / N * 1000
        _log(f"  Inference    : {elapsed_ms:.1f} ms/image  "
             f"({1000 / elapsed_ms:.1f} FPS)  [{device}]")

    _log("")
    return {"map50": map50, "map5095": map5095,
            "precision": prec, "recall": recall,
            "inference_ms": elapsed_ms}


def plot_training_curves(run_name: str) -> None:
    results_csv = RUNS_DIR / run_name / "results.csv"
    if not results_csv.exists():
        return

    rows: list[dict] = []
    with open(results_csv) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()}
                for r in csv.DictReader(f)]

    def _col(name: str) -> list[float] | None:
        for row in rows:
            for k in row:
                if name in k:
                    try:
                        return [float(r[k]) for r in rows]
                    except ValueError:
                        return None
        return None

    epochs_x   = list(range(1, len(rows) + 1))
    box_loss   = _col("train/box_loss")
    map50_hist = _col("metrics/mAP50")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if box_loss:
        axes[0].plot(epochs_x, box_loss, color="steelblue", linewidth=2)
        axes[0].set_title("Training Box Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        sns.despine(ax=axes[0])
    if map50_hist:
        best_val = max(map50_hist)
        axes[1].plot(epochs_x, map50_hist, color="darkorange", linewidth=2)
        axes[1].axhline(y=best_val, color="gray", linestyle="--",
                        alpha=0.7, label=f"Best: {best_val:.3f}")
        axes[1].set_title("Validation mAP@0.5")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP@0.5")
        axes[1].legend()
        sns.despine(ax=axes[1])
    plt.suptitle("InspectAI — Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RUNS_DIR / run_name / "training_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    _log(f"  ✓ Training curves → {out}")


# ── Stage 5: Export ───────────────────────────────────────────────────────────

def export_weights(best_weights: Path) -> Path:
    """Copy best.pt → models/inspectai_yolov8.pt (path detect.py expects)."""
    _log("── Stage 5: Export ─────────────────────────────────────────────")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / "inspectai_yolov8.pt"
    shutil.copy2(best_weights, dest)
    _log(f"  ✓ {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)\n")
    return dest


def smoke_test() -> None:
    """Confirm app.detect.run_detection() works with the new weights."""
    _log("── Smoke test ──────────────────────────────────────────────────")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import importlib
    import app.detect as detect_module
    detect_module._model = None
    importlib.reload(detect_module)
    from app.detect import run_detection

    test_img = None
    for lbl in sorted((DATA_YOLO / "labels" / "val").glob("*.txt")):
        if lbl.read_text().strip():
            p = DATA_YOLO / "images" / "val" / lbl.with_suffix(".jpg").name
            if p.exists():
                test_img = p
                break

    if not test_img:
        _log("  ⚠ No labelled val image found — skipping")
        return

    detections = run_detection(Image.open(test_img))
    _log(f"  Image: {test_img.name}")
    if detections:
        for d in detections:
            _log(f"  → {d.label:15s}  conf={d.confidence:.3f}  severity={d.severity}")
    else:
        _log("  No detections above threshold.")
    _log("")


def print_summary(metrics: dict[str, float], dest: Path,
                  n_defect: int, epochs: int) -> None:
    _log("── Results (paste into README.md) ──────────────────────────────")
    rows = [
        ("mAP@0.5",           f"{metrics['map50']:.3f}"),
        ("mAP@0.5:0.95",      f"{metrics['map5095']:.3f}"),
        ("Precision",         f"{metrics['precision']:.3f}"),
        ("Recall",            f"{metrics['recall']:.3f}"),
        ("Inference latency", f"{metrics['inference_ms']:.0f} ms/image"),
        ("Training images",   str(n_defect)),
        ("Base model",        "YOLOv8n (COCO pretrained)"),
        ("Training epochs",   str(epochs)),
    ]
    print("\n| Metric | Value |")
    print("|--------|-------|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    print(f"\nWeights : {dest}")
    print("\nNext steps:")
    print("  1. python main.py              → test the Gradio UI")
    print("  2. Update README.md with the table above")
    print("  3. Deploy to Hugging Face Spaces")
    print("  4. Start LangRobot (Project 2)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InspectAI YOLOv8 training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-build", action="store_true",
                   help="Skip dataset build (use existing data/yolo_dataset/)")
    p.add_argument("--epochs",    type=int, default=200)
    p.add_argument("--batch",     type=int, default=8,
                   help="Batch size (reduce to 4 if you hit OOM)")
    p.add_argument("--device",    type=str, default=None,
                   help="Device override: cpu | mps | 0  (auto-detected if omitted)")
    p.add_argument("--run-name",  type=str, default="inspectai_v1")
    p.add_argument("--seed",      type=int, default=42)

    # ── Fine-tune / continue training ────────────────────────────────────────
    p.add_argument(
        "--finetune", action="store_true",
        help="Continue training from existing weights instead of starting from scratch",
    )
    p.add_argument(
        "--finetune-weights", type=str,
        default="runs/inspectai_v1/weights/best.pt",
        help="Path to .pt checkpoint to fine-tune from (used with --finetune)",
    )
    p.add_argument(
        "--finetune-epochs", type=int, default=50,
        help="Number of additional epochs to run (used with --finetune)",
    )
    p.add_argument(
        "--min-delta", type=float, default=0.05,
        help=(
            "Minimum mAP@0.5 improvement required within --min-delta-patience epochs "
            "before training stops early.  E.g. 0.1 means the model must gain at least "
            "+0.10 mAP@0.5 in the window or training halts."
        ),
    )
    p.add_argument(
        "--min-delta-patience", type=int, default=30,
        help="Rolling window size (epochs) for the --min-delta early-stop check",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    _log("InspectAI — YOLOv8 Training Pipeline")
    _log(f"ROOT         : {ROOT}")
    _log(f"MVTec dataset: {DATA_RAW}")
    _log(f"YOLO dataset : {DATA_YOLO}\n")

    for d in (DATA_YOLO, MODEL_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Fine-tune mode wiring ─────────────────────────────────────────────────
    if args.finetune:
        finetune_weights = ROOT / args.finetune_weights
        # Default the run name to inspectai_v2 so artifacts don't overwrite v1
        if args.run_name == "inspectai_v1":
            args.run_name = "inspectai_v2"
        _log(f"Fine-tune mode  : {finetune_weights} → run '{args.run_name}'")
        _log(f"Extra epochs    : {args.finetune_epochs}")
        _log(f"Early stop      : mAP@0.5 must improve ≥{args.min_delta} "
             f"within {args.min_delta_patience} epochs\n")

    # Stage 1
    if args.finetune or args.skip_build:
        yaml_path = DATA_YOLO / "dataset.yaml"
        if not yaml_path.exists():
            _log("  ✗ dataset.yaml missing. Run without --finetune/--skip-build to rebuild.")
            sys.exit(1)
        label = "SKIPPED (--finetune)" if args.finetune else "SKIPPED (--skip-build)"
        _log(f"── Stage 1: Build dataset — {label}\n")
    else:
        yaml_path = build_dataset()

    # Stage 2
    visualise_dataset()

    # Stage 3
    device = detect_device(args.device)

    if args.finetune:
        best_weights = train(
            yaml_path, args.finetune_epochs, args.batch, device, args.run_name,
            finetune_weights = ROOT / args.finetune_weights,
            min_delta        = args.min_delta,
            min_delta_patience = args.min_delta_patience,
        )
    else:
        best_weights = train(yaml_path, args.epochs, args.batch,
                             device, args.run_name)

    # Stage 4
    metrics = evaluate(best_weights, yaml_path, device)
    plot_training_curves(args.run_name)

    # Stage 5
    dest = export_weights(best_weights)
    smoke_test()

    n_defect = sum(
        1 for lbl in (DATA_YOLO / "labels" / "train").glob("*.txt")
        if lbl.read_text().strip()
    ) + sum(
        1 for lbl in (DATA_YOLO / "labels" / "val").glob("*.txt")
        if lbl.read_text().strip()
    )
    print_summary(metrics, dest, n_defect, args.epochs)


if __name__ == "__main__":
    main()
