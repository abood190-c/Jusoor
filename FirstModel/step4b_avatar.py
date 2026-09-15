"""
STEP 4B: TEXT → SIGN AVATAR
=============================
Renders a hand skeleton avatar that animates through ASL fingerspelling
poses driven by typed text input.

How it works:
    1. User types text in the input box
    2. Each letter maps to a canonical landmark pose (from step4a)
    3. The avatar smoothly interpolates between poses as it steps
       through the letters
    4. The hand skeleton is drawn on a canvas using the same connection
       map as the live detector — reusing existing infrastructure

Interpolation:
    We use cosine interpolation (smoother than linear — eases in and out)
    between the current and next pose over TRANSITION_SECONDS.
    This makes the animation look like a hand naturally moving between
    signs rather than snapping between frozen poses.

Why cosine over linear interpolation:
    Linear interpolation moves at constant speed, which looks mechanical.
    Cosine interpolation starts slow, speeds up in the middle, then slows
    down — which matches how human hands actually move between poses.

Layout:
    Left panel  → text input + letter-by-letter playback controls
    Right panel → hand skeleton avatar

Usage:
    python step4b_avatar.py
"""

import cv2
import numpy as np
import pickle
import time

POSES_PATH  = "canonical_poses.pkl"

# Avatar display config
CANVAS_W    = 900
CANVAS_H    = 520
AVATAR_X    = 420       # where the avatar panel starts (x)
HAND_CENTER = (660, 280) # center point of the hand in the avatar panel
HAND_SCALE  = 120       # pixels per normalized unit — controls hand size

# Animation timing
HOLD_SECONDS       = 0.8   # how long to display each pose
TRANSITION_SECONDS = 0.3   # how long the interpolation between poses takes

# Colors
BG_COLOR      = (18,  18,  18)
PANEL_COLOR   = (28,  28,  28)
BONE_COLOR    = (0,   200, 100)
JOINT_COLOR   = (255, 255, 255)
ACCENT_COLOR  = (0,   180, 255)
TEXT_COLOR    = (220, 220, 220)
DIM_COLOR     = (100, 100, 100)
ACTIVE_COLOR  = (0,   230, 0)


# ─────────────────────────────────────────────
# LOAD CANONICAL POSES
# ─────────────────────────────────────────────

print(f"Loading canonical poses from {POSES_PATH}...")
with open(POSES_PATH, "rb") as f:
    canonical_poses = pickle.load(f)
print(f"✓ Loaded {len(canonical_poses)} poses: {sorted(canonical_poses.keys())}")


# ─────────────────────────────────────────────
# HAND SKELETON CONNECTIONS
# ─────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Finger tip indices for highlighting
FINGERTIPS = [4, 8, 12, 16, 20]


# ─────────────────────────────────────────────
# INTERPOLATION
# ─────────────────────────────────────────────

def cosine_interp(a, b, t):
    """
    Smoothly interpolate between pose vectors a and b.
    t=0 → a, t=1 → b. Uses cosine easing for natural motion.
    """
    smooth_t = (1 - np.cos(t * np.pi)) / 2
    return a * (1 - smooth_t) + b * smooth_t


def pose_to_points(pose_vec, center, scale):
    """
    Convert a 63-element normalized pose vector to 2D pixel coordinates
    for rendering. We use only x and y (drop z for 2D display).

    The z coordinate encodes depth — we'll use it to slightly scale
    joint sizes to give a subtle 3D feel, but not for position.
    """
    coords = pose_vec.reshape(21, 3)
    points = []
    for i in range(21):
        x =  coords[i, 0]   # positive x = right
        y = -coords[i, 1]   # flip y: in image coords y goes down, we want up
        px = int(center[0] + x * scale)
        py = int(center[1] + y * scale)
        points.append((px, py))
    return points


# ─────────────────────────────────────────────
# DRAW HAND
# ─────────────────────────────────────────────

def draw_hand(canvas, pose_vec, center, scale, alpha=1.0):
    """Draw the hand skeleton from a pose vector onto the canvas."""
    if pose_vec is None:
        return

    points = pose_to_points(pose_vec, center, scale)
    coords = pose_vec.reshape(21, 3)

    # Bones
    for a, b in HAND_CONNECTIONS:
        pt_a, pt_b = points[a], points[b]
        # Depth-based thickness: joints closer to camera (more negative z) thicker
        thickness = max(1, int(3 - coords[a, 2] * 2))
        color     = tuple(int(c * alpha) for c in BONE_COLOR)
        cv2.line(canvas, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    # Joints
    for i, pt in enumerate(points):
        depth     = coords[i, 2]
        radius    = 5 if i in FINGERTIPS else 3
        j_color   = tuple(int(c * alpha) for c in JOINT_COLOR)
        cv2.circle(canvas, pt, radius, j_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, radius, tuple(int(c * alpha) for c in BONE_COLOR), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# DRAW UI PANELS
# ─────────────────────────────────────────────

def draw_input_panel(canvas, input_text, sequence, current_idx, playing):
    """Left panel: text input and playback state."""
    # Panel background
    cv2.rectangle(canvas, (0, 0), (AVATAR_X - 10, CANVAS_H),
                  PANEL_COLOR, -1)

    # Title
    cv2.putText(canvas, "Text → Sign", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT_COLOR, 2)

    cv2.line(canvas, (20, 55), (AVATAR_X - 30, 55), (50, 50, 50), 1)

    # Input label
    cv2.putText(canvas, "Type your text:", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM_COLOR, 1)

    # Input box
    cv2.rectangle(canvas, (15, 100), (AVATAR_X - 25, 145),
                  (45, 45, 45), -1)
    cv2.rectangle(canvas, (15, 100), (AVATAR_X - 25, 145),
                  ACCENT_COLOR if not playing else DIM_COLOR, 1)

    display_text = input_text[-28:] if len(input_text) > 28 else input_text
    cv2.putText(canvas, display_text + ("|" if not playing else ""),
                (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 1)

    # Sequence display — show letters with current highlighted
    cv2.putText(canvas, "Sequence:", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM_COLOR, 1)

    x_pos = 20
    y_pos = 210
    for i, ch in enumerate(sequence[:18]):   # show up to 18 chars
        color     = ACTIVE_COLOR if i == current_idx else (
                    ACCENT_COLOR if i < current_idx else DIM_COLOR)
        size      = 0.9 if i == current_idx else 0.65
        thickness = 2 if i == current_idx else 1
        cv2.putText(canvas, ch.upper(), (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        x_pos += 22
        if x_pos > AVATAR_X - 40:
            x_pos  = 20
            y_pos += 30

    if len(sequence) > 18:
        cv2.putText(canvas, f"...+{len(sequence)-18} more",
                    (20, y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM_COLOR, 1)

    # Controls
    cv2.line(canvas, (20, CANVAS_H - 110), (AVATAR_X - 30, CANVAS_H - 110),
             (50, 50, 50), 1)
    controls = [
        ("ENTER",  "Play / Restart"),
        ("SPACE",  "Pause / Resume"),
        ("ESC",    "Clear input"),
        ("Q",      "Quit"),
    ]
    y = CANVAS_H - 95
    for key, desc in controls:
        cv2.putText(canvas, key, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_COLOR, 1)
        cv2.putText(canvas, f"  {desc}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM_COLOR, 1)
        y += 22


def draw_avatar_panel(canvas, current_letter):
    """Right panel background and label."""
    cv2.rectangle(canvas, (AVATAR_X, 0), (CANVAS_W, CANVAS_H),
                  BG_COLOR, -1)

    # Divider
    cv2.line(canvas, (AVATAR_X, 0), (AVATAR_X, CANVAS_H), (50, 50, 50), 1)

    # Current letter display
    if current_letter and current_letter != " ":
        cv2.putText(canvas, current_letter.upper(),
                    (CANVAS_W - 70, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, ACTIVE_COLOR, 3)


# ─────────────────────────────────────────────
# ANIMATION STATE
# ─────────────────────────────────────────────

class AvatarAnimator:
    def __init__(self):
        self.sequence      = []       # list of chars to animate
        self.current_idx   = -1
        self.playing       = False
        self.pose_start    = None
        self.from_pose     = np.zeros(63)
        self.to_pose       = np.zeros(63)
        self.current_pose  = np.zeros(63)
        self.phase         = "hold"   # "transition" or "hold"

    def start(self, text):
        """Begin animating through text."""
        # Filter to only chars we have poses for (letters + space)
        self.sequence    = [
            ch for ch in text.upper()
            if ch in canonical_poses or ch == " "
        ]
        self.current_idx = -1
        self.playing     = True
        self.phase       = "hold"
        self.advance()

    def advance(self):
        """Move to the next letter in the sequence."""
        self.current_idx += 1
        if self.current_idx >= len(self.sequence):
            self.playing = False
            return

        ch = self.sequence[self.current_idx]
        if ch == " ":
            # Space: show empty hand briefly
            self.to_pose = np.zeros(63)
        else:
            self.to_pose = canonical_poses.get(ch, np.zeros(63))

        self.from_pose  = self.current_pose.copy()
        self.pose_start = time.time()
        self.phase      = "transition"

    def update(self):
        """Call every frame. Returns current interpolated pose."""
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
            if elapsed >= HOLD_SECONDS:
                self.advance()

        return self.current_pose

    def toggle_pause(self):
        if self.sequence:
            self.playing = not self.playing
            if self.playing:
                self.pose_start = time.time()

    @property
    def current_letter(self):
        if 0 <= self.current_idx < len(self.sequence):
            return self.sequence[self.current_idx]
        return ""


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_avatar():
    animator   = AvatarAnimator()
    input_text = ""

    print("✓ Avatar ready. Type text and press ENTER to animate.\n")

    while True:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        # Update animation
        current_pose = animator.update()

        # Draw panels
        draw_input_panel(canvas, input_text,
                         animator.sequence, animator.current_idx,
                         animator.playing)
        draw_avatar_panel(canvas, animator.current_letter)

        # Draw hand
        draw_hand(canvas, current_pose, HAND_CENTER, HAND_SCALE)

        # Progress indicator
        if animator.playing and animator.sequence:
            progress = (animator.current_idx + 1) / len(animator.sequence)
            bar_w    = int((CANVAS_W - AVATAR_X - 20) * progress)
            cv2.rectangle(canvas,
                          (AVATAR_X + 10, CANVAS_H - 12),
                          (AVATAR_X + 10 + bar_w, CANVAS_H - 5),
                          ACCENT_COLOR, -1)
            cv2.rectangle(canvas,
                          (AVATAR_X + 10, CANVAS_H - 12),
                          (CANVAS_W - 10, CANVAS_H - 5),
                          DIM_COLOR, 1)

        cv2.imshow("Sign Language Avatar", canvas)

        key = cv2.waitKey(16) & 0xFF   # ~60fps

        if key == ord("q"):
            break

        elif key == 13:   # ENTER — start/restart animation
            if input_text.strip():
                animator.start(input_text)

        elif key == ord(" "):   # SPACE — pause/resume
            animator.toggle_pause()

        elif key == 27:   # ESC — clear input
            input_text = ""
            animator   = AvatarAnimator()

        elif key == 8:    # BACKSPACE
            input_text = input_text[:-1]

        elif 32 <= key <= 126:   # printable ASCII
            input_text += chr(key)

    cv2.destroyAllWindows()
    print("✓ Avatar closed.")


if __name__ == "__main__":
    run_avatar()