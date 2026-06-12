"""
test_detector.py — Frootify V2
================================
Quick sanity-check: opens the webcam, grabs one frame, runs the detector,
prints results, and saves an annotated image to  test_output.jpg

Run this BEFORE starting app.py to confirm:
  ✓ Model loads correctly
  ✓ Coordinate decoding is right (boxes appear over the tomatoes)
  ✓ Class labels are correct
  ✓ Confidence threshold is appropriate

Usage:
    python test_detector.py

Or with a still image instead of the webcam:
    python test_detector.py --image /path/to/tomato.jpg
"""

import sys
import cv2
import argparse
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))
import detector

MODEL_PATH = "models/best.onnx"


def test_with_frame(frame, session):
    print(f"\nFrame size: {frame.shape[1]}×{frame.shape[0]}")

    detections, rotten_found = detector.run_inference(session, frame)

    print(f"Detections found: {len(detections)}")
    for i, d in enumerate(detections):
        print(
            f"  [{i}] {d['class_name']:15s}  conf={d['confidence']:.3f}"
            f"  box={d['box']}"
        )

    print(f"\nRotten tomato present: {'YES → RED LED ON' if rotten_found else 'NO  → LED off'}")

    annotated = detector.draw_detections(frame.copy(), detections)
    out_path = "test_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {os.path.abspath(out_path)}")

    # Show window if a display is available
    try:
        cv2.imshow("Frootify V2 — Test", annotated)
        print("Press any key to close the window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("(No display available — check test_output.jpg instead)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a test image (optional)")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Copy best.onnx to models/best.onnx and try again.")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = detector.load_model(args.model)

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        frame = cv2.imread(args.image)
        if frame is None:
            print("ERROR: Could not read the image file.")
            sys.exit(1)
        print(f"Using image: {args.image}")
    else:
        print("Opening webcam (index 0) …")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("ERROR: Failed to grab frame from webcam.")
            sys.exit(1)
        print("Got frame from webcam.")

    test_with_frame(frame, session)


if __name__ == "__main__":
    main()
