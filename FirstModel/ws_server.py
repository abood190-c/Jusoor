"""
WEBSOCKET LANDMARK SERVER — CANONICAL POSES VERSION
=====================================================
Instead of streaming raw MediaPipe landmarks (which wobble),
this server:

  1. Loads canonical_poses.pkl at startup
  2. Sends the full poses table to the browser once on connect
  3. Every frame, runs the MLP classifier and sends only the
     predicted letter + confidence
  4. The browser looks up the canonical pose for that letter
     and drives the avatar — zero wobble, clean snapping

Usage:
    pip install websockets
    python ws_server.py

Then open hand_avatar_live.html in your browser and load your GLB.

Files needed in the same folder:
    sign_mlp_model.keras
    label_encoder.pkl
    canonical_poses.pkl
    hand_landmarker.task
    step1_extract_landmarks.py
"""

import asyncio
import json
import threading
import time
import cv2
import numpy as np
import pickle
import os
import urllib.request
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

from websockets.asyncio.server import serve

from step1_extract_landmarks import normalize_landmarks


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH      = "sign_mlp_model.keras"
ENCODER_PATH    = "label_encoder.pkl"
POSES_PATH      = "canonical_poses.pkl"
LANDMARKER_FILE = "hand_landmarker.task"
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

WS_HOST              = "localhost"
WS_PORT              = 8765
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_FRAMES     = 7


# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────

print("Loading model, encoder and canonical poses...")
classifier = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
with open(POSES_PATH, "rb") as f:
    canonical_poses_raw = pickle.load(f)
print(f"✓ Ready. Classes: {list(encoder.classes_)}")
print(f"✓ Canonical poses loaded: {len(canonical_poses_raw)} letters\n")


def pose_to_landmarks(flat_vec):
    """Convert flat 63-float vector to list of 21 {x,y,z} dicts."""
    coords = flat_vec.reshape(21, 3)
    return [
        {"x": float(coords[i, 0]),
         "y": float(coords[i, 1]),
         "z": float(coords[i, 2])}
        for i in range(21)
    ]


# Pre-serialise all poses once at startup — sent to browser on connect
POSES_JSON = {
    letter: pose_to_landmarks(vec)
    for letter, vec in canonical_poses_raw.items()
}

POSES_INIT_MSG = json.dumps({
    "type":  "poses_init",
    "poses": POSES_JSON,
})


# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

latest_landmarks = {"data": None}
landmark_lock    = threading.Lock()

connected_clients: set = set()
clients_lock = threading.Lock()

ws_loop: asyncio.AbstractEventLoop = None


# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────

def on_detection(result, output_image, timestamp_ms):
    with landmark_lock:
        latest_landmarks["data"] = (
            result.hand_landmarks[0] if result.hand_landmarks else None
        )


def build_landmarker():
    if not os.path.exists(LANDMARKER_FILE):
        print("Downloading hand landmarker model (~20MB)...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_FILE)
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_FILE)
    options = mp_vision.HandLandmarkerOptions(
        base_options                  = base_options,
        running_mode                  = mp_vision.RunningMode.LIVE_STREAM,
        num_hands                     = 1,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        result_callback               = on_detection,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# BROADCAST
# ─────────────────────────────────────────────

def broadcast(payload: dict):
    global ws_loop
    if ws_loop is None:
        return
    message = json.dumps(payload)
    asyncio.run_coroutine_threadsafe(_broadcast_async(message), ws_loop)


async def _broadcast_async(message: str):
    with clients_lock:
        clients = set(connected_clients)
    if not clients:
        return
    results = await asyncio.gather(
        *[client.send(message) for client in clients],
        return_exceptions=True
    )
    with clients_lock:
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                connected_clients.discard(client)


# ─────────────────────────────────────────────
# DETECTION LOOP (background thread)
# ─────────────────────────────────────────────

def detection_loop():
    cap        = cv2.VideoCapture(0)
    landmarker = build_landmarker()
    pred_buf   = deque(maxlen=SMOOTHING_FRAMES)
    frame_ts   = 0

    print("✓ Camera started.")
    print("  Open hand_avatar_live.html, load your GLB model.")
    print("  Press Q in this window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        landmarker.detect_async(mp_image, frame_ts)
        frame_ts += 33

        with landmark_lock:
            landmarks = latest_landmarks["data"]

        has_hand = landmarks is not None

        if has_hand:
            features   = normalize_landmarks(landmarks)
            features   = np.expand_dims(features, axis=0).astype(np.float32)
            raw_pred   = classifier(features, training=False).numpy()[0]
            class_id   = int(np.argmax(raw_pred))
            confidence = float(raw_pred[class_id])
            letter     = encoder.classes_[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:
                pred_buf.append(letter)
            else:
                pred_buf.append(None)

            valid          = [p for p in pred_buf if p is not None]
            display_letter = max(set(valid), key=valid.count) if valid else "nothing"

            # Send only the predicted letter — browser looks up pose itself
            broadcast({
                "type":       "prediction",
                "letter":     display_letter,
                "confidence": round(confidence, 3),
                "has_hand":   True,
            })

            color = (0, 230, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 100, 255)
            cv2.putText(frame,
                        f"{display_letter.upper()}  {confidence:.0%}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        else:
            pred_buf.clear()
            broadcast({"type": "prediction", "has_hand": False})
            cv2.putText(frame, "No hand", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

        cv2.imshow("WS Server — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(
            "WS Server — Q to quit", cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("✓ Detection loop closed.")


# ─────────────────────────────────────────────
# WEBSOCKET SERVER
# ─────────────────────────────────────────────

async def ws_handler(websocket):
    with clients_lock:
        connected_clients.add(websocket)
    addr = websocket.remote_address
    print(f"  [WS] Browser connected: {addr}")
    try:
        # Send poses table immediately on connect — browser caches it
        await websocket.send(POSES_INIT_MSG)
        print(f"  [WS] Poses table sent to {addr}")
        await websocket.wait_closed()
    finally:
        with clients_lock:
            connected_clients.discard(websocket)
        print(f"  [WS] Browser disconnected: {addr}")


async def start_ws_server():
    global ws_loop
    ws_loop = asyncio.get_event_loop()
    print(f"✓ WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
    async with serve(ws_handler, WS_HOST, WS_PORT):
        await asyncio.Future()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    detect_thread = threading.Thread(
        target=detection_loop,
        daemon=True,
    )
    detect_thread.start()

    try:
        asyncio.run(start_ws_server())
    except KeyboardInterrupt:
        print("\n✓ Server stopped.")