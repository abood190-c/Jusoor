"""
STEP 4A: COMPUTE CANONICAL POSES (FIXED)
==========================================
Previous version averaged across both original AND mirrored samples,
which caused X coordinates to cancel out to zero (symmetric around origin).

Fix: We take only the first half of each class — the original samples
recorded before mirror augmentation was applied. Since mirror_augmentation.py
concatenated [original, mirrored] and then shuffled, we can't easily
separate them by position anymore.

Better approach: re-extract canonical poses directly from the dataset
images for one consistent hand orientation, OR use the pre-augmentation
CSV if you saved it.

Simplest reliable fix: re-run landmark extraction on a small sample
per class (50 images) at the original orientation, compute means from
those. This is fast (~1 min) and guaranteed to give one-handed poses.

Usage:
    python step4a_compute_poses.py
"""

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from step1_extract_landmarks import normalize_landmarks

DATASET_DIR     = "dataset/train"
LANDMARKER_FILE = "hand_landmarker.task"
OUTPUT_PATH     = "canonical_poses.pkl"
SAMPLES_PER_CLASS = 100   # enough for a stable mean, fast to process


def build_landmarker(model_file):
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


def compute_canonical_poses():
    class_names = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    landmarker      = build_landmarker(LANDMARKER_FILE)
    canonical_poses = {}

    print(f"Computing canonical poses from dataset images ({SAMPLES_PER_CLASS} per class)...\n")

    for class_name in class_names:

        # nothing is synthetic — zero vector by definition
        if class_name.lower() == "nothing":
            canonical_poses[class_name] = np.zeros(63)
            print(f"  {class_name:<10} → zero vector (synthetic class)")
            continue

        class_dir   = os.path.join(DATASET_DIR, class_name)
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:SAMPLES_PER_CLASS]

        samples = []

        for img_file in image_files:
            img_bgr = cv2.imread(os.path.join(class_dir, img_file))
            if img_bgr is None:
                continue

            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                continue

            features = normalize_landmarks(result.hand_landmarks[0])
            samples.append(features)

        if not samples:
            print(f"  {class_name:<10} → no samples detected, using zeros")
            canonical_poses[class_name] = np.zeros(63)
            continue

        mean_pose               = np.mean(samples, axis=0)
        canonical_poses[class_name] = mean_pose
        print(f"  {class_name:<10} {len(samples):>4} samples used → mean pose computed")

    landmarker.close()

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(canonical_poses, f)

    print(f"\n✓ Saved {len(canonical_poses)} canonical poses to {OUTPUT_PATH}")
    print("Now run: python step4b_avatar.py")


if __name__ == "__main__":
    compute_canonical_poses()