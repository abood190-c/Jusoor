"""
STEP 5: LIVE DETECTOR WITH HAND TRACKING
==========================================
Updated for MediaPipe 0.10+ task-based API.

Uses LIVE_STREAM running mode with a callback — this is the correct mode
for real-time webcam input. It's asynchronous: MediaPipe calls our
callback function whenever a result is ready, so the main loop never blocks.

Requirements:
    - label_encoder.py         (from step2_train_mlp.py)
    - hand_landmarker.task      (downloaded by step1, auto-downloaded here too)
    - step1_extract_landmarks.py in the same folder (for normalize_landmarks)

Usage:
    python step3_live_detector.py
    Press Q to quit.
"""

"""
SIGN LANGUAGE DETECTOR — WITH WORD ACCUMULATOR
================================================
Extends the live detector with a word accumulator layer that:
  1. Commits a letter after it's held confidently for HOLD_SECONDS
  2. Inserts a space after SPACE_SECONDS of no detected hand or 'space' sign
  3. Deletes the last letter when 'del' is signed and held
  4. Accumulates letters into words and words into a sentence buffer
  5. Shows a live text display on screen

Letter commitment logic:
    We track how many consecutive frames the SAME letter has been
    predicted above the confidence threshold. Once that duration
    exceeds HOLD_SECONDS, the letter commits and a cooldown starts
    so the same letter isn't immediately re-committed.

    Why cooldown instead of requiring a return to 'nothing':
        In practice users don't always return to a neutral pose
        between letters. A fixed cooldown (0.8s) is more reliable
        than waiting for a specific neutral state, especially given
        M/N/O ambiguity in the current model.

Usage:
    python step3_live_detector.py
    Press Q to quit, BACKSPACE to manually delete last char.
"""

import cv2
import numpy as np
import pickle
import time
import threading
import urllib.request
import os
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

from step1_extract_landmarks import normalize_landmarks


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH           = "sign_mlp_model.keras"
ENCODER_PATH         = "label_encoder.pkl"
LANDMARKER_FILE      = "hand_landmarker.task"
LANDMARKER_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_FRAMES     = 7

# Word accumulator timing
HOLD_SECONDS         = 1.0    # hold duration to commit a letter
COOLDOWN_SECONDS     = 0.8    # gap between commits
SPACE_SECONDS        = 2.0    # silence before auto-space
MAX_SENTENCE_LEN     = 80     # chars shown in sentence display


# ─────────────────────────────────────────────
# LOAD MODEL AND ENCODER
# ─────────────────────────────────────────────

print("Loading model and label encoder...")
classifier = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
print(f"✓ Ready. Classes: {list(encoder.classes_)}")


# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
# This section handles a threading problem specific to LIVE_STREAM mode.
# latest_result is a shared dictionary acting as a mailbox between the two threads.
# result_lock is a threading lock that prevents both threads from accessing
# latest_result at the exact same moment, which could corrupt the data.
latest_result = {"landmarks": None, "timestamp": 0}
result_lock   = threading.Lock()

# on_detection_result is a callback function —  MediaPipe calls it automatically
# whenever it finishes processing a frame. It just stores the latest landmarks
# into the shared mailbox.
def on_detection_result(result, output_image, timestamp_ms):
    with result_lock:
        latest_result["landmarks"] = (
            result.hand_landmarks[0] if result.hand_landmarks else None
        )
        latest_result["timestamp"] = timestamp_ms


# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────

def build_live_landmarker(model_file):
    base_options = mp_python.BaseOptions(model_asset_path=model_file)
    options      = mp_vision.HandLandmarkerOptions(
        base_options                  = base_options,
        running_mode                  = mp_vision.RunningMode.LIVE_STREAM,
        num_hands                     = 1,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        result_callback               = on_detection_result,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# The fw, fh parameters are frame width and height

def draw_hand_skeleton(frame, landmarks, fw, fh):
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 0), 1)

def get_bounding_box(landmarks, fw, fh, pad=20):
    xs = [lm.x * fw for lm in landmarks]
    ys = [lm.y * fh for lm in landmarks]
    return (
        max(0,  int(min(xs)) - pad),
        max(0,  int(min(ys)) - pad),
        min(fw, int(max(xs)) + pad),
        min(fh, int(max(ys)) + pad),
    )

def draw_text_panel(frame, sentence, current_word, hold_progress,
                    current_letter, fh, fw):
    panel_h = 110
    panel_y = fh - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (fw, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, panel_y), (fw, panel_y), (80, 80, 80), 1)

    # Current letter + hold progress bar
    if current_letter and current_letter not in ("nothing", "space", "del"):
        cv2.putText(frame, current_letter.upper(),
                    (15, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 230, 0), 3)
        cv2.rectangle(frame, (60, panel_y + 55), (180, panel_y + 65),
                      (80, 80, 80), -1)
        if hold_progress > 0:
            bar_x2 = 60 + int(120 * hold_progress)
            color  = (0, 200, 255) if hold_progress < 1.0 else (0, 255, 0)
            cv2.rectangle(frame, (60, panel_y + 55),
                          (bar_x2, panel_y + 65), color, -1)

    # Current word
    cv2.putText(frame, "Word: " + (current_word if current_word else "_"),
                (200, panel_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Sentence
    display = sentence[-MAX_SENTENCE_LEN:] if sentence else ""
    cv2.putText(frame, display,
                (10, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Instructions
    cv2.putText(frame, "Q=quit  BKSP=delete",
                (fw - 200, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


# ─────────────────────────────────────────────
# WORD ACCUMULATOR
# ─────────────────────────────────────────────

class WordAccumulator:
    """
    State machine for letter commitment.

    IDLE     → waiting for a stable sign
    HOLDING  → sign detected, counting hold time
    COOLDOWN → letter just committed, ignoring input briefly
    """

    def __init__(self):
        self.sentence       = ""
        self.current_word   = ""
        self.state          = "IDLE"
        self.hold_letter    = None
        self.hold_start     = None
        self.cooldown_start = None
        self.last_hand_time = time.time()

    def update(self, letter, has_hand):
        now           = time.time()
        committed     = None
        hold_progress = 0.0

        if not has_hand:
            self.hold_letter = None
            self.hold_start  = None
            self.state       = "IDLE"
            if now - self.last_hand_time > SPACE_SECONDS:
                if self.current_word:
                    self.sentence    += self.current_word + " "
                    self.current_word = ""
                self.last_hand_time = now
            return committed, hold_progress

        self.last_hand_time = now

        if self.state == "COOLDOWN":
            if now - self.cooldown_start >= COOLDOWN_SECONDS:
                self.state = "IDLE"
            else:
                return committed, hold_progress

        if letter == "nothing":
            self.hold_letter = None
            self.hold_start  = None
            self.state       = "IDLE"
            return committed, hold_progress

        # space and del — hold to trigger
        if letter in ("space", "del"):
            if self.hold_letter != letter:
                self.hold_letter = letter
                self.hold_start  = now
                self.state       = "HOLDING"
            elapsed       = now - self.hold_start
            hold_progress = min(elapsed / HOLD_SECONDS, 1.0)
            if elapsed >= HOLD_SECONDS:
                if letter == "space":
                    if self.current_word:
                        self.sentence    += self.current_word + " "
                        self.current_word = ""
                else:  # del
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                    elif self.sentence:
                        self.sentence = self.sentence.rstrip()
                        self.sentence = self.sentence[:self.sentence.rfind(" ") + 1]
                committed           = letter
                self.state          = "COOLDOWN"
                self.cooldown_start = now
                self.hold_letter    = None
            return committed, hold_progress

        # Regular letter
        if self.hold_letter != letter:
            self.hold_letter = letter
            self.hold_start  = now
            self.state       = "HOLDING"

        elapsed       = now - self.hold_start
        hold_progress = min(elapsed / HOLD_SECONDS, 1.0)

        if elapsed >= HOLD_SECONDS:
            self.current_word   += letter.upper()
            committed            = letter
            self.state           = "COOLDOWN"
            self.cooldown_start  = now
            self.hold_letter     = None

        return committed, hold_progress

    def manual_delete(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence.rstrip()
            last_space    = self.sentence.rfind(" ")
            self.sentence = self.sentence[:last_space + 1]

    @property
    def display_sentence(self):
        return self.sentence + self.current_word


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_detector():
    if not os.path.exists(LANDMARKER_FILE):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_FILE)

    cap         = cv2.VideoCapture(0)
    landmarker  = build_live_landmarker(LANDMARKER_FILE)
    pred_buf    = deque(maxlen=SMOOTHING_FRAMES)
    accumulator = WordAccumulator()
    frame_ts    = 0
    max_failures = 30  # about 1 second at 30fps
    print("✓ Webcam started. Press Q to quit, BACKSPACE to delete.\n")

    consecutive_failures = 0


    while True:
        ret, frame = cap.read()
        if frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("Camera feed lost. Exiting.")
                break
            continue
        consecutive_failures = 0  # reset on success

        frame    = cv2.flip(frame, 1)
        fh, fw   = frame.shape[:2]

        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        landmarker.detect_async(mp_image, frame_ts)
        frame_ts += 33

        with result_lock:
            landmarks = latest_result["landmarks"]

        has_hand       = landmarks is not None
        display_letter = None
        confidence     = 0.0

        if has_hand:
            draw_hand_skeleton(frame, landmarks, fw, fh)
            x1, y1, x2, y2 = get_bounding_box(landmarks, fw, fh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)

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

            valid = [p for p in pred_buf if p is not None]
            display_letter = max(set(valid), key=valid.count) if valid else "nothing"

            color = (0, 230, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 100, 255)
            cv2.putText(frame, display_letter.upper(),
                        (x1, max(y1 - 15, 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
            cv2.putText(frame, f"{confidence:.0%}",
                        (x1, max(y1 - 15, 50) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            pred_buf.clear()
            cv2.putText(frame, "No hand detected",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 100, 255), 2)

        committed, hold_progress = accumulator.update(
            display_letter or "nothing", has_hand
        )

        # Brief green flash on commit
        if committed and committed not in ("del",):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 255, 100), -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        draw_text_panel(
            frame,
            sentence       = accumulator.sentence,
            current_word   = accumulator.current_word,
            hold_progress  = hold_progress,
            current_letter = display_letter,
            fh             = fh,
            fw             = fw,
        )

        cv2.imshow("Sign Language Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 8:  # BACKSPACE
            accumulator.manual_delete()

        if cv2.getWindowProperty("Sign Language Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"\nFinal text: {accumulator.display_sentence}")
    print("✓ Detector closed.")


if __name__ == "__main__":
    run_detector()