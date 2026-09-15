"""
STEP 1 & 2: LANDMARK EXTRACTION + NORMALIZATION
================================================
Updated for MediaPipe 0.10+ which replaced mp.solutions.hands
with the new task-based API (mp.tasks.vision.HandLandmarker).

Usage:
    python step1_extract_landmarks.py
"""

import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_DIR  = "dataset/train"
OUTPUT_CSV   = "landmarks.csv"
MODEL_FILE   = "hand_landmarker.task"
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


# ─────────────────────────────────────────────
# DOWNLOAD MEDIAPIPE MODEL IF NEEDED
# ─────────────────────────────────────────────
# The new Tasks API requires a .task model file downloaded separately.
# Auto-download it on the first run so you don't have to do it manually.

def download_model_if_needed(model_file, url):
    if not os.path.exists(model_file):
        print(f"Downloading MediaPipe hand landmarker model (~20MB)...")
        urllib.request.urlretrieve(url, model_file)
        print(f"✓ Downloaded to {model_file}")
    else:
        print(f"✓ Found model file: {model_file}")


# ─────────────────────────────────────────────
# NORMALIZATION FUNCTION
# ─────────────────────────────────────────────
# CRITICAL: This function is imported by the live detector.
# Never copy-paste it — always import from this file to guarantee
# training and inference use identical preprocessing.

def normalize_landmarks(landmarks):
    """
    Convert raw MediaPipe landmarks to a scale- and position-invariant vector.

    Why position-invariant (subtract wrist):
        The same sign at different positions on screen produces different
        raw coordinates. Subtracting landmark 0 (wrist) makes all points
        relative to the hand's own position.

    Why scale-invariant (divide by wrist-to-mid-base distance):
        A hand close to the camera looks larger. Dividing by the distance
        between wrist (0) and middle finger base (9) normalizes hand size.

    Input:  list of 21 landmark objects with .x .y .z attributes
    Output: numpy array of shape (63,) — flattened [x,y,z] * 21 points
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Step 1: Translate — wrist becomes origin
    wrist  = coords[0].copy()
    coords = coords - wrist

    # Step 2: Scale by wrist-to-middle-finger-base distance
    scale  = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale

    return coords.flatten()


# ─────────────────────────────────────────────
# BUILD HAND LANDMARKER
# ─────────────────────────────────────────────

def build_hand_landmarker(model_file):
    """
    Why IMAGE running mode:
        We're processing individual still images from the dataset,
        not a continuous video stream. IMAGE mode runs full detection
        independently on each frame — no temporal smoothing applied,
        which is correct for static dataset images.
        The live detector uses LIVE_STREAM mode instead.
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_file)
    options      = mp_vision.HandLandmarkerOptions(
        base_options                  = base_options,
        running_mode                  = mp_vision.RunningMode.IMAGE,
        num_hands                     = 1,
        min_hand_detection_confidence = 0.3,
        min_hand_presence_confidence  = 0.3,
        min_tracking_confidence       = 0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# EXTRACTION LOOP
# ─────────────────────────────────────────────

def extract_landmarks_from_dataset(dataset_dir, output_csv, model_file):
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    print(f"\nFound {len(class_names)} classes: {class_names}")

    skipped   = 0
    processed = 0

    landmarker = build_hand_landmarker(model_file)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

        for class_name in class_names:
            class_dir   = os.path.join(dataset_dir, class_name)
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            print(f"\nProcessing '{class_name}' — {len(image_files)} images")

            for img_file in tqdm(image_files, desc=class_name):
                img_bgr = cv2.imread(os.path.join(class_dir, img_file))
                if img_bgr is None:
                    skipped += 1
                    continue

                # Convert BGR → RGB (because MediaPipe expects RGB)
                img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Wrap in MediaPipe Image container (required by new API)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

                result   = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skipped += 1
                    continue

                features = normalize_landmarks(result.hand_landmarks[0])
                writer.writerow([class_name] + features.tolist())
                processed += 1

    landmarker.close()

    total     = processed + skipped
    skip_rate = (skipped / total * 100) if total > 0 else 0
    print(f"\n✓ Extraction complete.")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped} (no hand detected) — {skip_rate:.1f}%")
    if skip_rate > 30:
        print("  ⚠ High skip rate. Check dataset image quality or lower confidence thresholds.")
    print(f"✓ Saved to  : {output_csv}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    download_model_if_needed(MODEL_FILE, MODEL_URL)
    extract_landmarks_from_dataset(DATASET_DIR, OUTPUT_CSV, MODEL_FILE)