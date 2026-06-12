"""
detector.py — Frootify V2
ONNX-based YOLOv8n tomato detector for Raspberry Pi 4.

Model output shape: (1, 6, 8400)
Each detection vector: [x_center, y_center, width, height, fresh_score, rotten_score]
Coordinates are in the 640×640 input space.
"""

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE   = 640          # model expects 640×640
CONF_THRESH  = 0.60         # confirmed from your Pi test (rotten ~0.79, bg ~0.34)
IOU_THRESH   = 0.45         # NMS overlap threshold
CLASS_NAMES  = ["Fresh Tomato", "Rotten Tomato"]

# BGR colours for bounding boxes and labels
COLOUR_FRESH  = (0,  200,  0)    # green
COLOUR_ROTTEN = (0,   0, 220)    # red

LABEL_BG_ALPHA = 0.6             # label background opacity


def load_model(model_path: str):
    """Load the ONNX model and return the InferenceSession."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime not installed. Run: pip install onnxruntime"
        )
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],   # Pi 4 has no GPU
    )
    print(f"[Detector] Model loaded: {model_path}")
    print(f"[Detector] Input  : {session.get_inputs()[0].name}  {session.get_inputs()[0].shape}")
    print(f"[Detector] Output : {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    return session


def _preprocess(frame: np.ndarray):
    """
    Resize + normalise a BGR frame into a (1, 3, 640, 640) float32 tensor.
    Returns (tensor, scale_x, scale_y, pad_x, pad_y) for coordinate recovery.

    Uses letterbox padding so the aspect ratio is preserved, which matches
    how Ultralytics YOLOv8 pre-processes images during export.
    """
    h, w = frame.shape[:2]

    # Scale keeping aspect ratio
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding to reach 640×640
    pad_top    = (INPUT_SIZE - new_h) // 2
    pad_bottom = INPUT_SIZE - new_h - pad_top
    pad_left   = (INPUT_SIZE - new_w) // 2
    pad_right  = INPUT_SIZE - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    return tensor, scale, pad_left, pad_top


def _decode(raw_output: np.ndarray, scale: float, pad_x: int, pad_y: int,
            orig_w: int, orig_h: int):
    """
    Decode raw ONNX output into a list of detections.

    raw_output shape: (1, 6, 8400)
      axis-1 rows : [x_c, y_c, w, h, fresh_score, rotten_score]
      axis-2 cols : 8400 candidate anchor boxes

    Returns list of dicts:
      { 'box': [x1,y1,x2,y2],   ← pixel coords in original frame
        'class_id': int,
        'class_name': str,
        'confidence': float }
    """
    preds = raw_output[0]               # shape (6, 8400)

    x_c   = preds[0]
    y_c   = preds[1]
    bw    = preds[2]
    bh    = preds[3]
    scores_fresh  = preds[4]
    scores_rotten = preds[5]

    # Best class per anchor
    class_scores = np.stack([scores_fresh, scores_rotten], axis=0)  # (2, 8400)
    class_ids    = np.argmax(class_scores, axis=0)                   # (8400,)
    confidences  = np.max(class_scores, axis=0)                      # (8400,)

    # Filter by confidence threshold
    mask = confidences >= CONF_THRESH
    if not np.any(mask):
        return []

    x_c   = x_c[mask]
    y_c   = y_c[mask]
    bw    = bw[mask]
    bh    = bh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    # Convert from 640-space → original frame space
    # 1. Remove letterbox padding offset
    x_c = (x_c - pad_x) / scale
    y_c = (y_c - pad_y) / scale
    bw  = bw  / scale
    bh  = bh  / scale

    # 2. xywh → xyxy
    x1 = np.clip(x_c - bw / 2, 0, orig_w)
    y1 = np.clip(y_c - bh / 2, 0, orig_h)
    x2 = np.clip(x_c + bw / 2, 0, orig_w)
    y2 = np.clip(y_c + bh / 2, 0, orig_h)

    # NMS per class
    detections = []
    for cls in range(len(CLASS_NAMES)):
        cls_mask = class_ids == cls
        if not np.any(cls_mask):
            continue

        boxes_cls  = np.stack([x1[cls_mask], y1[cls_mask],
                                x2[cls_mask], y2[cls_mask]], axis=1).astype(np.float32)
        confs_cls  = confidences[cls_mask].astype(np.float32)

        # cv2.dnn.NMSBoxes expects [x, y, w, h] format
        cv2_boxes = [[int(b[0]), int(b[1]),
                      int(b[2] - b[0]), int(b[3] - b[1])]
                     for b in boxes_cls]

        indices = cv2.dnn.NMSBoxes(
            cv2_boxes, confs_cls.tolist(), CONF_THRESH, IOU_THRESH
        )

        if len(indices) == 0:
            continue

        for i in indices.flatten():
            detections.append({
                "box"        : [int(x1[cls_mask][i]),
                                int(y1[cls_mask][i]),
                                int(x2[cls_mask][i]),
                                int(y2[cls_mask][i])],
                "class_id"   : cls,
                "class_name" : CLASS_NAMES[cls],
                "confidence" : float(confs_cls[i]),
            })

    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame in-place.
    Returns the annotated frame.
    """
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        label  = f"{det['class_name']}  {det['confidence']:.2f}"
        colour = COLOUR_FRESH if cls_id == 0 else COLOUR_ROTTEN

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        lbl_y1 = max(y1 - th - baseline - 6, 0)
        lbl_y2 = max(y1, th + baseline + 6)
        cv2.rectangle(overlay, (x1, lbl_y1), (x1 + tw + 6, lbl_y2), colour, -1)

        # Label text
        cv2.putText(
            overlay, label,
            (x1 + 3, lbl_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend overlay with original for semi-transparent label backgrounds
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, frame, 1 - LABEL_BG_ALPHA, 0, frame)
    return frame


def run_inference(session, frame: np.ndarray):
    """
    Full pipeline: preprocess → infer → decode → return detections.

    Returns:
        detections  : list of detection dicts (may be empty)
        rotten_found: bool — True if at least one rotten tomato was detected
    """
    orig_h, orig_w = frame.shape[:2]

    tensor, scale, pad_x, pad_y = _preprocess(frame)

    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: tensor})[0]

    detections = _decode(raw, scale, pad_x, pad_y, orig_w, orig_h)

    rotten_found = any(d["class_id"] == 1 for d in detections)
    return detections, rotten_found
