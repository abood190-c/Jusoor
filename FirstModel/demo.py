"""
BIDIRECTIONAL SIGN LANGUAGE TRANSLATOR
========================================
Single window demo with two modes toggled by TAB:

  MODE A — Sign → Text  (default)
      Webcam feed with hand tracking and word accumulator.
      Signs are detected, letters committed, words and sentences built.

  MODE B — Text → Sign
      Type text using the keyboard.
      Avatar animates through the ASL fingerspelling poses.

Controls:
    TAB         Toggle between Mode A and Mode B
    Q           Quit
    BACKSPACE   Delete last character (both modes)
    ENTER       (Mode B) Play / restart animation
    SPACE       (Mode B) Pause / resume animation
    ESC         (Mode B) Clear input text

Layout (both modes):
    Left panel  → mode-specific input/output info + controls
    Right panel → webcam feed (Mode A) or avatar (Mode B)
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

MODEL_PATH      = "sign_mlp_model.keras"
ENCODER_PATH    = "label_encoder.pkl"
POSES_PATH      = "canonical_poses.pkl"
LANDMARKER_FILE = "hand_landmarker.task"
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Window
WIN_W, WIN_H    = 1280, 600
PANEL_W         = 340       # left info panel width
CAM_W           = WIN_W - PANEL_W
CAM_H           = WIN_H

# Detector config
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_FRAMES     = 7
HOLD_SECONDS         = 1.0
COOLDOWN_SECONDS     = 0.8
SPACE_SECONDS        = 2.0

# Avatar config
HAND_CENTER          = (int(CAM_W * 0.55) + PANEL_W, WIN_H // 2 + 30)
HAND_SCALE           = 120
HOLD_SECONDS_AVATAR  = 0.8
TRANSITION_SECONDS   = 0.3

# Colors
C_BG        = (15,  15,  15)
C_PANEL     = (25,  25,  25)
C_DIVIDER   = (55,  55,  55)
C_ACCENT    = (0,   180, 255)
C_GREEN     = (0,   220, 100)
C_ORANGE    = (0,   165, 255)
C_WHITE     = (230, 230, 230)
C_DIM       = (100, 100, 100)
C_RED       = (60,  60,  220)
C_MODE_A    = (0,   210, 120)   # green tint for Sign→Text
C_MODE_B    = (180, 100, 255)   # purple tint for Text→Sign


# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────

print("Loading model, encoder and poses...")
classifier = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
with open(POSES_PATH, "rb") as f:
    canonical_poses = pickle.load(f)
print(f"✓ Ready.")


# ─────────────────────────────────────────────
# SHARED MEDIAPIPE STATE
# ─────────────────────────────────────────────

latest_landmarks = {"data": None}
landmark_lock    = threading.Lock()


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
    options      = mp_vision.HandLandmarkerOptions(
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
# HAND DRAWING
# ─────────────────────────────────────────────

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGERTIPS = [4, 8, 12, 16, 20]


def draw_skeleton_on_frame(frame, landmarks, fw, fh):
    pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], C_GREEN, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 5 if i in FINGERTIPS else 3
        cv2.circle(frame, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, C_GREEN, 1, cv2.LINE_AA)


def draw_skeleton_avatar(canvas, pose_vec, center, scale):
    if pose_vec is None:
        return
    coords = pose_vec.reshape(21, 3)
    pts    = []
    for i in range(21):
        px = int(center[0] + coords[i, 0] * scale)
        py = int(center[1] - coords[i, 1] * scale)
        pts.append((px, py))
    for a, b in CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], C_MODE_B, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 6 if i in FINGERTIPS else 4
        cv2.circle(canvas, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, C_MODE_B, 1, cv2.LINE_AA)


def get_bbox(landmarks, fw, fh, pad=20):
    xs = [lm.x * fw for lm in landmarks]
    ys = [lm.y * fh for lm in landmarks]
    return (max(0, int(min(xs))-pad), max(0, int(min(ys))-pad),
            min(fw, int(max(xs))+pad), min(fh, int(max(ys))+pad))


# ─────────────────────────────────────────────
# WORD ACCUMULATOR
# ─────────────────────────────────────────────

class WordAccumulator:
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
                else:
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
            self.sentence = self.sentence[:self.sentence.rfind(" ") + 1]

    @property
    def display_text(self):
        return self.sentence + self.current_word


# ─────────────────────────────────────────────
# AVATAR ANIMATOR
# ─────────────────────────────────────────────

def cosine_interp(a, b, t):
    s = (1 - np.cos(t * np.pi)) / 2
    return a * (1 - s) + b * s


class AvatarAnimator:
    def __init__(self):
        self.sequence     = []
        self.current_idx  = -1
        self.playing      = False
        self.pose_start   = None
        self.from_pose    = np.zeros(63)
        self.to_pose      = np.zeros(63)
        self.current_pose = np.zeros(63)
        self.phase        = "hold"

    def start(self, text):
        self.sequence    = [
            ch for ch in text.upper()
            if ch in canonical_poses or ch == " "
        ]
        self.current_idx = -1
        self.playing     = True
        self.advance()

    def advance(self):
        self.current_idx += 1
        if self.current_idx >= len(self.sequence):
            self.current_idx = len(self.sequence) - 1  # clamp, never overshoot
            self.playing     = False
            return
        ch             = self.sequence[self.current_idx]
        self.to_pose   = canonical_poses.get(ch, np.zeros(63)) if ch != " " else np.zeros(63)
        self.from_pose = self.current_pose.copy()
        self.pose_start= time.time()
        self.phase     = "transition"

    def update(self):
        if not self.playing or self.current_idx < 0:
            return self.current_pose
        now     = time.time()
        elapsed = now - self.pose_start
        if self.phase == "transition":
            t = min(elapsed / TRANSITION_SECONDS, 1.0)
            self.current_pose = cosine_interp(self.from_pose, self.to_pose, t)
            if elapsed >= TRANSITION_SECONDS:
                self.current_pose = self.to_pose.copy()
                self.pose_start   = now
                self.phase        = "hold"
        elif self.phase == "hold":
            if elapsed >= HOLD_SECONDS_AVATAR:
                self.advance()
        return self.current_pose

    def toggle_pause(self):
        # Don't allow resume if animation already finished
        if not self.sequence or (not self.playing and self.current_idx >= len(self.sequence) - 1 and self.phase == "hold"):
            return
        self.playing = not self.playing
        if self.playing:
            self.pose_start = time.time()

    def reset(self):
        self.__init__()

    @property
    def current_letter(self):
        if 0 <= self.current_idx < len(self.sequence):
            return self.sequence[self.current_idx]
        return ""

    @property
    def progress(self):
        if not self.sequence:
            return 0.0
        return min((self.current_idx + 1) / len(self.sequence), 1.0)


# ─────────────────────────────────────────────
# UI DRAWING
# ─────────────────────────────────────────────

def draw_panel(canvas, mode, accumulator, animator, input_text):
    """Draw the left info panel for the current mode."""
    cv2.rectangle(canvas, (0, 0), (PANEL_W, WIN_H), C_PANEL, -1)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, WIN_H), C_DIVIDER, 1)

    # Mode badge
    mode_color = C_MODE_A if mode == "A" else C_MODE_B
    mode_label = "Sign  →  Text" if mode == "A" else "Text  →  Sign"
    cv2.rectangle(canvas, (10, 10), (PANEL_W - 10, 48), mode_color, -1)
    cv2.putText(canvas, mode_label, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # TAB hint
    cv2.putText(canvas, "TAB = switch mode", (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)

    cv2.line(canvas, (10, 78), (PANEL_W - 10, 78), C_DIVIDER, 1)

    if mode == "A":
        # ── Sign → Text panel ────────────────────────────────────────
        cv2.putText(canvas, "Output Text:", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1)

        # Sentence box
        cv2.rectangle(canvas, (10, 112), (PANEL_W - 10, 210),
                      (35, 35, 35), -1)
        cv2.rectangle(canvas, (10, 112), (PANEL_W - 10, 210),
                      C_MODE_A, 1)

        # Word wrap the sentence text into the box
        text  = accumulator.display_text
        words = (text[-120:] if len(text) > 120 else text)
        lines = []
        line  = ""
        for ch in words:
            line += ch
            if len(line) >= 20:
                lines.append(line)
                line = ""
        if line:
            lines.append(line)
        for i, ln in enumerate(lines[-4:]):
            cv2.putText(canvas, ln, (18, 138 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_WHITE, 1)

        # Current word highlight
        cv2.putText(canvas, "Current word:", (15, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_DIM, 1)
        cv2.putText(canvas, accumulator.current_word or "_",
                    (15, 255), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, C_MODE_A, 2)

        # Controls
        cv2.line(canvas, (10, WIN_H - 130), (PANEL_W - 10, WIN_H - 130),
                 C_DIVIDER, 1)
        controls = [
            ("Hold 1s",   "Commit letter"),
            ("del sign",  "Delete letter"),
            ("space sign","Add space"),
            ("BKSP key",  "Manual delete"),
            ("Q",         "Quit"),
        ]
        y = WIN_H - 118
        for k, v in controls:
            cv2.putText(canvas, f"{k:<10} {v}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)
            y += 20

    else:
        # ── Text → Sign panel ─────────────────────────────────────────
        cv2.putText(canvas, "Type your text:", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1)

        # Input box
        cv2.rectangle(canvas, (10, 112), (PANEL_W - 10, 155),
                      (35, 35, 35), -1)
        border_color = C_MODE_B if not animator.playing else C_DIM
        cv2.rectangle(canvas, (10, 112), (PANEL_W - 10, 155),
                      border_color, 1)
        disp = input_text[-22:] if len(input_text) > 22 else input_text
        cv2.putText(canvas, disp + ("|" if not animator.playing else ""),
                    (18, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 1)

        # Sequence strip
        cv2.putText(canvas, "Sequence:", (15, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_DIM, 1)
        x_pos, y_pos = 15, 200
        for i, ch in enumerate(animator.sequence[:40]):
            if i == animator.current_idx:
                color, thick = C_MODE_B, 2
            elif i < animator.current_idx:
                color, thick = C_DIM, 1
            else:
                color, thick = C_WHITE, 1
            cv2.putText(canvas, ch, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick)
            x_pos += 16
            if x_pos > PANEL_W - 25:
                x_pos  = 15
                y_pos += 22
                if y_pos > WIN_H - 160:
                    break

        # Progress bar
        if animator.sequence:
            bx1, by1 = 10, WIN_H - 155
            bx2, by2 = PANEL_W - 10, WIN_H - 143
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
            fill = bx1 + int((bx2 - bx1) * animator.progress)
            cv2.rectangle(canvas, (bx1, by1), (fill, by2), C_MODE_B, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), C_DIM, 1)

        # Controls
        cv2.line(canvas, (10, WIN_H - 132), (PANEL_W - 10, WIN_H - 132),
                 C_DIVIDER, 1)
        controls = [
            ("ENTER",  "Play / restart"),
            ("SPACE",  "Pause / resume"),
            ("BKSP",   "Delete char"),
            ("ESC",    "Clear all"),
            ("Q",      "Quit"),
        ]
        y = WIN_H - 120
        for k, v in controls:
            cv2.putText(canvas, f"{k:<8} {v}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)
            y += 20


def draw_mode_a(canvas, frame, landmarks, display_letter,
                confidence, hold_progress, accumulator):
    """Render the Sign→Text webcam view into the right panel."""
    fh, fw = frame.shape[:2]

    # Scale and paste webcam feed into right panel
    cam_region = cv2.resize(frame, (CAM_W, CAM_H))
    canvas[0:CAM_H, PANEL_W:WIN_W] = cam_region

    if landmarks:
        # Scale landmark coords to the resized frame
        scale_x = CAM_W / fw
        scale_y = CAM_H / fh

        class _ScaledLM:
            def __init__(self, lm):
                self.x = lm.x
                self.y = lm.y

        scaled = [_ScaledLM(lm) for lm in landmarks]
        pts    = [
            (int(PANEL_W + lm.x * CAM_W), int(lm.y * CAM_H))
            for lm in landmarks
        ]

        # Draw skeleton directly on canvas in the cam region
        for a, b in CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], C_GREEN, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            r = 5 if i in FINGERTIPS else 3
            cv2.circle(canvas, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r, C_GREEN, 1, cv2.LINE_AA)

        # Bounding box
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x1 = max(PANEL_W, min(xs) - 20)
        y1 = max(0,       min(ys) - 20)
        x2 = min(WIN_W,   max(xs) + 20)
        y2 = min(WIN_H,   max(ys) + 20)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_GREEN, 2)

        if display_letter and display_letter != "nothing":
            color = C_GREEN if confidence >= CONFIDENCE_THRESHOLD else C_RED
            cv2.putText(canvas, display_letter.upper(),
                        (x1, max(y1 - 15, 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

            # Hold progress arc indicator
            if hold_progress > 0:
                bar_len = int((x2 - x1) * hold_progress)
                bar_col = (0, 200, 255) if hold_progress < 1.0 else C_GREEN
                cv2.rectangle(canvas, (x1, y2 + 6),
                              (x1 + bar_len, y2 + 14), bar_col, -1)
                cv2.rectangle(canvas, (x1, y2 + 6),
                              (x2, y2 + 14), C_DIM, 1)
    else:
        cv2.putText(canvas, "No hand detected",
                    (PANEL_W + 20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2)


def draw_mode_b(canvas, pose_vec, animator, input_text):
    """Render the Text→Sign avatar into the right panel."""
    # Dark background for avatar area
    cv2.rectangle(canvas, (PANEL_W, 0), (WIN_W, WIN_H), C_BG, -1)

    # Current letter large display
    if animator.current_letter and animator.current_letter != " ":
        cv2.putText(canvas, animator.current_letter,
                    (WIN_W - 80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_MODE_B, 3)

    # Draw avatar hand
    draw_skeleton_avatar(canvas, pose_vec, HAND_CENTER, HAND_SCALE)

    # Status
    if not animator.playing and animator.sequence:
        status = "PAUSED" if animator.current_idx >= 0 else "DONE"
        cv2.putText(canvas, status,
                    (PANEL_W + 20, WIN_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_DIM, 1)
    elif not animator.sequence:
        cv2.putText(canvas, "Type text and press ENTER",
                    (PANEL_W + 40, WIN_H // 2 + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_DIM, 1)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cap         = cv2.VideoCapture(0)
    landmarker  = build_landmarker()
    pred_buf    = deque(maxlen=SMOOTHING_FRAMES)
    accumulator = WordAccumulator()
    animator    = AvatarAnimator()

    mode        = "A"         # "A" = Sign→Text, "B" = Text→Sign
    input_text  = ""
    frame_ts    = 0

    cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Translator", WIN_W, WIN_H)

    print("✓ Running. TAB to switch modes, Q to quit.\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame = cv2.flip(raw_frame, 1)
        fh, fw    = raw_frame.shape[:2]

        # Always run MediaPipe (needed in mode A, kept warm in mode B)
        img_rgb  = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        landmarker.detect_async(mp_image, frame_ts)
        frame_ts += 33

        with landmark_lock:
            landmarks = latest_landmarks["data"]

        # ── Build canvas ──────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)

        if mode == "A":
            # Classify
            has_hand       = landmarks is not None
            display_letter = None
            confidence     = 0.0
            hold_progress  = 0.0

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

                valid = [p for p in pred_buf if p is not None]
                display_letter = max(set(valid), key=valid.count) if valid else "nothing"
            else:
                pred_buf.clear()

            committed, hold_progress = accumulator.update(
                display_letter or "nothing", has_hand
            )

            # Flash on commit
            if committed and committed not in ("del",):
                overlay = canvas.copy()
                cv2.rectangle(overlay, (PANEL_W, 0), (WIN_W, WIN_H),
                              (0, 255, 100), -1)
                cv2.addWeighted(overlay, 0.07, canvas, 0.93, 0, canvas)

            draw_mode_a(canvas, raw_frame, landmarks, display_letter,
                        confidence, hold_progress, accumulator)

        else:
            # Mode B — update avatar
            pose = animator.update()
            draw_mode_b(canvas, pose, animator, input_text)

        # Always draw the left panel on top
        draw_panel(canvas, mode, accumulator, animator, input_text)

        cv2.imshow("Sign Language Translator", canvas)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(16) & 0xFF

        if key == ord("q"):
            break

        elif key == 9:   # TAB — toggle mode
            mode = "B" if mode == "A" else "A"
            pred_buf.clear()

        elif key == 8:   # BACKSPACE
            if mode == "A":
                accumulator.manual_delete()
            else:
                input_text = input_text[:-1]

        elif mode == "B":
            if key == 13:    # ENTER
                if input_text.strip():
                    animator.start(input_text)
            elif key == ord(" "):
                animator.toggle_pause()
            elif key == 27:  # ESC
                input_text = ""
                animator.reset()
            elif 32 <= key <= 126:
                input_text += chr(key)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    if mode == "A":
        print(f"\nFinal text: {accumulator.display_text}")
    print("✓ Closed.")


if __name__ == "__main__":
    main()