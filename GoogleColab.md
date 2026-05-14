# 🖐️ YOLOv8n Hand Detection — Google Colab Workflow

This document describes the full training pipeline implemented in [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), designed to run on **Google Colab** with a free T4 GPU.

---

## Prerequisites

- A **Google account** (for Colab + Google Drive)
- A **free Roboflow account** to get an API key → https://app.roboflow.com
- GPU enabled in Colab: `Runtime → Change runtime type → T4 GPU`

---

## Workflow Overview

```
Install libraries → Mount Drive → Download dataset → Train → Evaluate → Export ONNX
      Step 1           Step 2         Step 3          Step 4    Step 5      Step 6
```

---

## Step-by-Step Description

### STEP 1 — Install Required Libraries
Installs all Python dependencies needed for training and export:
- **ultralytics** — YOLOv8 framework
- **roboflow** — dataset download API
- **onnx / onnxruntime** — model export and inference
- **opencv-python** — image processing

Also verifies that a CUDA GPU is available. If not, the user is prompted to enable it.

---

### STEP 2 — Mount Google Drive
Mounts Google Drive at `/content/drive` so all files persist across sessions.

Creates the following project folders inside `MyDrive/hand_detection/`:

| Folder | Purpose |
|--------|---------|
| `dataset/` | Downloaded training data |
| `runs/` | Training outputs (weights, metrics, plots) |
| `models/` | Final exported model files |

> ⚡ A **Restore Paths** cell is also provided. After any Colab reconnect, re-run Steps 1, 2, and the restore cell to recover all path variables without re-downloading data.

---

### STEP 3 — Download Hand Dataset from Roboflow
Downloads the **EgoHands Public** dataset (version 9) from Roboflow Universe:
- ~4800 images, pre-labeled with a single class: `hand`
- Format: YOLOv8 (images + `.txt` label files + `data.yaml`)
- Saved directly to Google Drive

**Dataset source:** https://universe.roboflow.com/brad-dwyer/egohands-public

The `DATA_YAML` path variable is set automatically after download.

---

### STEP 4 — Train YOLOv8n with Transfer Learning
Loads `yolov8n.pt` (pretrained on COCO, ~6 MB) and fine-tunes it on the hand dataset.

**Training configuration:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 50 | Sufficient for convergence on this dataset |
| `imgsz` | 320 | Small size → faster training, suitable for edge deployment |
| `batch` | 16 | Fits T4 GPU VRAM |
| `lr0` | 0.01 | Standard initial learning rate |
| `patience` | 15 | Early stopping if no improvement |
| `device` | 0 | Use GPU |

Results and best weights are saved to `runs/hand_yolov8n/` on Google Drive.  
Estimated training time: **15–30 minutes** on a free T4 GPU.

---

### STEP 5 — Evaluate the Model
Runs validation on the held-out validation set and reports:

| Metric | Target |
|--------|--------|
| mAP50 | > 0.80 ✅ |
| mAP50-95 | > 0.50 ✅ |
| Precision | — |
| Recall | — |

Provides a recommendation based on the achieved mAP50 score.

---

### STEP 6 — Export to ONNX Format
Converts `best.pt` (PyTorch) → `best.onnx` (ONNX) for cross-platform inference.

**Export settings:**

| Parameter | Value |
|-----------|-------|
| `format` | `onnx` |
| `imgsz` | 320 (matches training) |
| `opset` | 12 (widely supported) |
| `dynamic` | False (fixed input, better for edge) |

The exported `best.onnx` is copied to `models/best.onnx` on Google Drive.

**To download the model:**
1. Open the left sidebar in Colab → Files icon
2. Navigate to `/content/drive/MyDrive/hand_detection/models/`
3. Right-click `best.onnx` → **Download**

---

## Output Files

After the full pipeline completes, the following files are available in your Google Drive:

```
MyDrive/hand_detection/
├── dataset/
│   └── egohands-public-9/      ← training data (images + labels)
├── runs/
│   └── hand_yolov8n/
│       └── weights/
│           ├── best.pt          ← best PyTorch checkpoint
│           └── last.pt          ← last epoch checkpoint
└── models/
    └── best.onnx                ← final model for deployment
```

---

## Using the Model After Export

The `best.onnx` file can be used for inference on:
- **Laptop** — using `onnxruntime` (`scripts/test_inference.py`)
- **Raspberry Pi** — using `onnxruntime` (CPU-only, optimized by `opset=12` + `imgsz=320`)
