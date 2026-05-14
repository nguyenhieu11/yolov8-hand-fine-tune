"""
test_inference.py
==================
Test the trained ONNX model on your laptop.

Usage:
  # Test with webcam (default):
  python scripts/test_inference.py

  # Test with a single image:
  python scripts/test_inference.py --source path/to/image.jpg

  # Test with a folder of images:
  python scripts/test_inference.py --source path/to/images/

Requirements:
  pip install -r requirements.txt
  Place your best.onnx in the project root (yolov8-hand-fine-tune/best.onnx)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ─────────────────────────────────────────────────────────
# CONFIG — adjust if needed
MODEL_PATH  = Path(__file__).resolve().parent.parent / "best.onnx"
INPUT_SIZE  = 320       # must match the imgsz used during export
CONF_THRESH = 0.8      # minimum confidence to show a detection
IOU_THRESH  = 0.6      # NMS IoU threshold
CLASS_NAMES = ["hand"]
BOX_COLOR   = (0, 255, 0)   # green bounding box
TEXT_COLOR  = (0, 0, 0)     # black text
# ─────────────────────────────────────────────────────────


def load_model(model_path: Path) -> ort.InferenceSession:
    """Load ONNX model with onnxruntime."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found: {model_path}\n"
            "   Download best.onnx from Google Drive after training\n"
            "   and place it in the project root folder."
        )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    print(f"✅ Model loaded: {model_path.name}")
    print(f"   Input  : {session.get_inputs()[0].shape}")
    print(f"   Output : {session.get_outputs()[0].shape}")
    return session


def preprocess(frame: np.ndarray, size: int):
    """
    Resize + normalize a BGR frame for YOLOv8 inference.
    Returns: (blob, scale_x, scale_y, pad_x, pad_y)
    """
    h, w = frame.shape[:2]

    # Letterbox resize (keep aspect ratio, pad the rest)
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Create padded canvas
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR → RGB, HWC → CHW, normalize to [0, 1]
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # add batch dim

    return blob, scale, pad_x, pad_y


def postprocess(outputs: np.ndarray, orig_w: int, orig_h: int,
                scale: float, pad_x: int, pad_y: int,
                conf_thresh: float, iou_thresh: float):
    """
    Parse YOLOv8 output tensor → list of (x1, y1, x2, y2, conf, class_id).
    YOLOv8 output shape: [1, 5+nc, num_anchors]  (cx, cy, w, h, conf per class)
    """
    preds = outputs[0]          # shape: [1, 5, num_anchors]
    preds = preds[0].T          # shape: [num_anchors, 5]

    boxes, scores = [], []
    for pred in preds:
        cx, cy, bw, bh = pred[:4]
        conf = pred[4]
        if conf < conf_thresh:
            continue

        # Convert from padded-letterboxed coords back to original image coords
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        x1 = max(0, min(int(x1), orig_w))
        y1 = max(0, min(int(y1), orig_h))
        x2 = max(0, min(int(x2), orig_w))
        y2 = max(0, min(int(y2), orig_h))

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    if not boxes:
        return []

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        [[x, y, x2 - x, y2 - y] for (x, y, x2, y2) in boxes],
        scores, conf_thresh, iou_thresh
    )
    detections = []
    for i in np.array(indices).flatten():
        x1, y1, x2, y2 = boxes[i]
        detections.append((x1, y1, x2, y2, scores[i], 0))
    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    return frame


def run_webcam(session: ort.InferenceSession):
    """Run real-time inference on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (camera index 0).")
        print("   Try: --source 1 or --source 2 for other cameras.")
        return

    input_name = session.get_inputs()[0].name
    print("\n🎥 Webcam started. Press 'q' to quit.")

    fps_counter, fps_time = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = preprocess(frame, INPUT_SIZE)
        outputs = session.run(None, {input_name: blob})
        detections = postprocess(outputs, w, h, scale, pad_x, pad_y,
                                 CONF_THRESH, IOU_THRESH)
        frame = draw_detections(frame, detections)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_counter, fps_time = 0, time.time()
        else:
            fps = fps_counter / max(time.time() - fps_time, 1e-9)

        cv2.putText(frame, f"FPS: {fps:.1f} | Hands: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Hand Detection (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_images(session: ort.InferenceSession, source: Path):
    """Run inference on a single image or folder of images."""
    input_name = session.get_inputs()[0].name

    if source.is_file():
        image_paths = [source]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [p for p in source.iterdir() if p.suffix.lower() in exts]

    print(f"\n🖼️  Running on {len(image_paths)} image(s)...")

    for img_path in sorted(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"   ⚠️  Could not read: {img_path.name}")
            continue

        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = preprocess(frame, INPUT_SIZE)
        outputs = session.run(None, {input_name: blob})
        detections = postprocess(outputs, w, h, scale, pad_x, pad_y,
                                 CONF_THRESH, IOU_THRESH)
        frame = draw_detections(frame, detections)

        print(f"   {img_path.name}: {len(detections)} hand(s) detected")

        # Resize for display (keep it within 900px wide)
        disp_w = min(w, 900)
        disp_h = int(h * disp_w / w)
        display = cv2.resize(frame, (disp_w, disp_h))
        cv2.imshow(f"{img_path.name} — press any key", display)
        key = cv2.waitKey(0)
        if key == 27:   # ESC to stop early
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Hand Detection Inference (ONNX)")
    parser.add_argument(
        "--source", type=str, default="webcam",
        help="'webcam' | path to image | path to image folder"
    )
    parser.add_argument(
        "--model", type=str, default=str(MODEL_PATH),
        help="Path to best.onnx"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help="Confidence threshold (default: 0.5)"
    )
    args = parser.parse_args()

    session = load_model(Path(args.model))

    if args.source == "webcam":
        run_webcam(session)
    else:
        run_images(session, Path(args.source))


if __name__ == "__main__":
    main()
