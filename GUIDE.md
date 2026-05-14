# 🖐️ Hand Detection with YOLOv8n — Complete Step-by-Step Guide

**Stack:** YOLOv8n · ONNX Runtime · Google Colab (free) · Roboflow (free)  
**Goal:** Train a hand detection model on Google Colab → test on laptop → deploy on Raspberry Pi + ROS2 Jazzy later

---

## 📁 Project Structure

```
yolov8-hand-fine-tune/
├── GUIDE.md                         ← You are here
├── GoogleColab.md                   ← Colab workflow reference
├── requirements.txt                 ← Laptop dependencies (for testing only)
├── best.onnx                        ← Place here after downloading from Colab
├── scripts/
│   └── test_inference.py            ← Test ONNX model on your laptop
└── notebooks/
    └── train_colab.ipynb            ← Main training notebook (run this on Colab)
```

> ℹ️ **All training happens on Google Colab.** The dataset is downloaded directly inside
> Colab. Your laptop is only used for testing the final exported `best.onnx` model.

---

## PHASE 1 — Setup Your Laptop (for testing only)

> Do this once. Your laptop is only used for testing the final ONNX model after training.

### Step 1 — Create a Virtual Environment (Recommended)

A virtual environment keeps all project packages **isolated** from your system Python.

```bash
cd /home/hieu/learning/yolov8-hand-fine-tune

# Create virtual environment named "venv" inside the project folder
python3 -m venv venv
```

**Activate it** (you must do this every time you open a new terminal):
```bash
source venv/bin/activate
```

Your terminal prompt will change to show `(venv)` — this means it is active:
```
(venv) hieu@pc:~/learning/yolov8-hand-fine-tune$
```

**To deactivate** when you are done working:
```bash
deactivate
```

> 💡 **Tip:** Always make sure `(venv)` is visible in your terminal before running
> any `python` or `pip` command in this project.

### Step 2 — Install Dependencies

Make sure the virtual environment is active (see Step 1), then:

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import onnxruntime, cv2; print('All OK')"
```

> ⚠️ The `venv/` folder is large. It is local only —
> never upload it to Google Drive or commit it to Git.

---

## PHASE 2 — Get a Free Roboflow API Key

> You need this to download the free pre-labeled dataset inside Colab.

### Step 1 — Create Roboflow Account

1. Go to **https://app.roboflow.com** → Sign up free (use Google or email)
2. After login → click your **avatar** (top-right corner)
3. Click **"Settings"** → click **"Roboflow API"** tab
4. Copy your **Private API Key** (looks like: `aBcD1234eFgH5678`)
5. **Save it** — you will need it in Colab Step 3 (Download Dataset)

---

## PHASE 3 — Train on Google Colab

> Google Colab gives you a free GPU (T4) in the cloud.
> Your laptop does NOT need a GPU.

### Step 3 — Open Google Colab

1. Go to **https://colab.research.google.com**
2. Click **File → Upload notebook**
3. Upload the file: `notebooks/train_colab.ipynb`

### Step 4 — Enable Free GPU

1. In Colab top menu: **Runtime → Change runtime type**
2. Under "Hardware accelerator" → select **T4 GPU**
3. Click **Save**

You should see `T4` in the top-right corner of Colab.

### Step 5 — Run Cell 1: Install Libraries

- Click the first code cell (Install Required Libraries)
- Click the ▶ button or press **Shift+Enter**
- Wait for it to finish (~2 minutes)
- ✅ You should see: `GPU detected: Tesla T4`

> If you see ❌ No GPU: go back to Step 4.

### Step 6 — Run Cell 2: Mount Google Drive

- Run the cell
- A link will appear → **click it**
- Choose your Google account → click **Allow**
- Copy the authorization code → **paste it** in the input box → press Enter
- ✅ You should see: `Mounted at /content/drive`

> This saves all your training results to Google Drive permanently.
> Even if Colab disconnects, your files are safe.

### Step 7 — Run Cell 3: Download Dataset

1. Open `notebooks/train_colab.ipynb` in a text editor or Colab
2. Find the line: `ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"`
3. Replace `YOUR_API_KEY_HERE` with your key from **Step 2**
4. Run the cell
- ✅ You should see: `~3600 images downloaded` split into train/val/test

### Step 8 — Run Cell 4: Train the Model

- Run the cell
- ⏳ Training takes **15–30 minutes** on free T4 GPU
- You will see a progress bar with loss values updating each epoch
- ✅ When done, you should see `mAP50: 0.8xx`

> **If Colab disconnects during training:**
> - Re-run cells 1 → 2 → 4 (skip dataset download, it's already saved)
> - Training will resume from the last checkpoint

### Step 9 — Run Cell 5: Evaluate Model

- Run the cell
- Check the output:
  - `mAP50 > 0.80` → ✅ Ready to export
  - `mAP50 0.65–0.80` → ⚠️ Acceptable, can improve later
  - `mAP50 < 0.65` → ❌ Try increasing epochs to 100

### Step 10 — Run Cell 6: Export to ONNX

- Run the cell
- ✅ `best.onnx` is saved to your Google Drive at:
  `/MyDrive/hand_detection/models/best.onnx`

### Step 11 — Download ONNX Model to Laptop

1. In Colab left sidebar → click the **📁 folder icon**
2. Navigate to: `drive → MyDrive → hand_detection → models`
3. Right-click **`best.onnx`** → **Download**
4. Move the downloaded file to: `/home/hieu/learning/yolov8-hand-fine-tune/best.onnx`

---

## PHASE 4 — Test on Your Laptop

### Step 12 — Test with Webcam

```bash
cd /home/hieu/learning/yolov8-hand-fine-tune
python scripts/test_inference.py
```

- A window will open showing your webcam feed
- Hold up your hand → you should see a green bounding box
- Press **`q`** to quit

### Step 13 — Test with Images

```bash
# Single image
python scripts/test_inference.py --source path/to/image.jpg

# Folder of images
python scripts/test_inference.py --source path/to/folder/

# Adjust confidence threshold (default 0.5)
python scripts/test_inference.py --conf 0.4
```

---

## ✅ Done! What You Have Now

| File | Description |
|------|-------------|
| `best.onnx` | Trained hand detector, runs anywhere with ONNX Runtime |
| `scripts/test_inference.py` | Real-time webcam / image test |

---

## PHASE 5 — Next Steps (Future)

- [ ] Copy `best.onnx` to Raspberry Pi
- [ ] Install `onnxruntime` on RPi
- [ ] Wrap inference in a **ROS2 Jazzy node**
- [ ] Subscribe to `/camera/image_raw`, publish to `/hand_detections`
- [ ] Fine-tune with images from RPi camera if accuracy is low

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` on laptop | Virtual env not active — run `source venv/bin/activate` first |
| `(venv)` missing from terminal | Re-activate: `cd yolov8-hand-fine-tune && source venv/bin/activate` |
| Colab disconnects during training | Re-run cells 1→2, then jump to Cell 4 (Drive keeps your data) |
| `No GPU detected` in Colab | Runtime → Change runtime type → T4 GPU |
| `API key invalid` | Re-copy from roboflow.com → Settings → API |
| `best.onnx not found` on laptop | Make sure you placed it in `yolov8-hand-fine-tune/best.onnx` |
| Webcam not opening | Try `--source 1` or `--source 2` |
| Low FPS on laptop | Normal for CPU, target is RPi performance later |
| `mAP50 < 0.65` | Increase `epochs=100` in Colab Cell 4 and retrain |
