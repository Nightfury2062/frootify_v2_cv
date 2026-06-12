"""
app.py — Frootify V2  (main entry point)
=========================================
Fully autonomous tomato monitoring system for Raspberry Pi 4.

Start with:
    python app.py

Stop with:
    Ctrl + C   or close the display window (press Q)

Directory layout expected:
    frootify_v2/
    ├── app.py
    ├── detector.py
    ├── gpio_manager.py
    ├── env_sensor.py
    └── models/
        └── best.onnx        ← your exported model
"""

import cv2
import time
import signal
import sys
import os
import threading
import logging
from datetime import datetime

import detector
import gpio_manager as gpio
import env_sensor   as env

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH      = "models/best.onnx"
CAMERA_INDEX    = 0          # 0 = first USB webcam
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
ENV_CHECK_EVERY = 3600       # seconds between DHT reads (1 hour)
SHOW_DISPLAY    = True       # set False on headless Pi to skip cv2.imshow

LOG_DIR         = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── Logging setup ──────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, f"frootify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("frootify")


# ── Graceful shutdown ──────────────────────────────────────────────────────────
_running = True

def _shutdown(sig=None, frame=None):
    global _running
    log.info("Shutdown signal received. Stopping …")
    _running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Overlay helpers ────────────────────────────────────────────────────────────

def _draw_status_bar(frame, rotten_found, temp, humid, env_alert, fps):
    """Render a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 36
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)

    # Rotten status
    rot_colour = (0, 0, 220) if rotten_found else (0, 180, 0)
    rot_text   = "ROTTEN DETECTED" if rotten_found else "All Fresh"
    cv2.putText(frame, rot_text, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rot_colour, 2, cv2.LINE_AA)

    # Env status
    env_text = ""
    if temp is not None:
        env_text = f"Temp:{temp:.1f}C  Hum:{humid:.0f}%"
        if env_alert:
            env_text += "  ENV WARN"
    env_colour = (0, 140, 255) if env_alert else (160, 160, 160)
    cv2.putText(frame, env_text, (w // 2 - 100, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, env_colour, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.1f} fps", (w - 90, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1, cv2.LINE_AA)


# ── Environmental sensor thread ────────────────────────────────────────────────

def _env_loop():
    """
    Background thread: reads the DHT sensor once, then sleeps for
    ENV_CHECK_EVERY seconds, then reads again — forever.
    """
    log.info("[EnvThread] Starting. First read in 5 seconds …")
    time.sleep(5)   # let the camera loop start first

    while _running:
        temp, humid, alert = env.read_now()
        gpio.set_env_led(alert)
        if alert:
            log.warning(
                f"[EnvThread] ENVIRONMENTAL ALERT  "
                f"Temp={temp}°C  Humidity={humid}%"
            )
        else:
            log.info(
                f"[EnvThread] OK  Temp={temp}°C  Humidity={humid}%"
            )

        # Sleep in small increments so we can respond to _running=False quickly
        for _ in range(ENV_CHECK_EVERY):
            if not _running:
                break
            time.sleep(1)

    log.info("[EnvThread] Exiting.")


# ── Camera loop ────────────────────────────────────────────────────────────────

def _camera_loop(session):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error(f"Cannot open camera index {CAMERA_INDEX}. Check the connection.")
        _shutdown()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # minimise latency on Pi

    log.info(f"[Camera] Opened at {FRAME_WIDTH}×{FRAME_HEIGHT}.")

    # State tracking to avoid spamming the log
    prev_rotten_state = None
    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0

    while _running:
        ret, frame = cap.read()
        if not ret:
            log.warning("[Camera] Frame grab failed — retrying …")
            time.sleep(0.1)
            continue

        # ── Inference ──────────────────────────────────────────────────────────
        detections, rotten_found = detector.run_inference(session, frame)

        # ── LED control ────────────────────────────────────────────────────────
        gpio.set_rotten_led(rotten_found)

        # ── Log state changes only (avoid flood) ───────────────────────────────
        if rotten_found != prev_rotten_state:
            if rotten_found:
                rotten_count = sum(1 for d in detections if d["class_id"] == 1)
                log.warning(
                    f"[Vision] ROTTEN DETECTED — {rotten_count} rotten tomato(s) in frame."
                )
            else:
                log.info("[Vision] Frame clear — no rotten tomatoes.")
            prev_rotten_state = rotten_found

        # ── Annotate frame ─────────────────────────────────────────────────────
        detector.draw_detections(frame, detections)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 2.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer   = time.time()

        # ── Status bar ─────────────────────────────────────────────────────────
        temp, humid, env_alert = env.get_last_reading()
        _draw_status_bar(frame, rotten_found, temp, humid, env_alert, fps)

        # ── Display ────────────────────────────────────────────────────────────
        if SHOW_DISPLAY:
            cv2.imshow("Frootify V2 — Tomato Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or Esc
                _shutdown()
                break

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    log.info("[Camera] Released.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Frootify V2  —  Tomato Quality Monitor")
    log.info("=" * 60)

    # LED self-test
    gpio.test_leds()

    # Load model
    if not os.path.exists(MODEL_PATH):
        log.error(
            f"Model not found at: {MODEL_PATH}\n"
            f"Copy your best.onnx to:  {os.path.abspath(MODEL_PATH)}"
        )
        sys.exit(1)

    log.info(f"Loading model from {MODEL_PATH} …")
    session = detector.load_model(MODEL_PATH)
    log.info("Model loaded. Starting subsystems …")

    # Start environmental monitoring in background
    env_thread = threading.Thread(target=_env_loop, daemon=True, name="EnvThread")
    env_thread.start()

    # Run camera loop in main thread (cv2.imshow requires main thread on some Pi setups)
    try:
        _camera_loop(session)
    finally:
        env.cleanup()
        gpio.cleanup()
        log.info("Frootify V2 stopped cleanly.")


if __name__ == "__main__":
    main()
