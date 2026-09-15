"""
DIAGNOSTIC: Skip Rate Analysis
================================
Run this to see exactly which classes are being skipped and why,
before deciding whether to proceed to training.

Usage:
    python diagnose_skips.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

DATASET_DIR  = "dataset/train"
LANDMARKER_FILE = "hand_landmarker.task"
SAMPLE_PER_CLASS = 50  # check first 50 images per class for speed


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


def analyze(dataset_dir, model_file, sample_per_class):
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    landmarker = build_landmarker(model_file)
    results    = {}

    print(f"\nSampling up to {sample_per_class} images per class...\n")

    for class_name in class_names:
        class_dir   = os.path.join(dataset_dir, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:sample_per_class]

        detected = 0
        skipped  = 0

        for img_file in image_files:
            img_bgr = cv2.imread(os.path.join(class_dir, img_file))
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

            if result.hand_landmarks:
                detected += 1
            else:
                skipped += 1

        total              = detected + skipped
        skip_rate          = skipped / total * 100 if total > 0 else 0
        results[class_name]= (detected, skipped, skip_rate)

    landmarker.close()

    # ── Print report ──────────────────────────────────────────────────
    print(f"{'Class':<10} {'Detected':>10} {'Skipped':>10} {'Skip %':>10}  Verdict")
    print("─" * 60)

    high_skip_classes    = []
    expected_skip_classes= []  # nothing/space/delete — expected to skip

    for class_name, (detected, skipped, skip_rate) in sorted(results.items()):
        verdict = ""
        if class_name.lower() in ["nothing", "space", "del", "delete", "blank"]:
            verdict = "✓ expected (non-hand class)"
            expected_skip_classes.append(class_name)
        elif skip_rate > 40:
            verdict = "⚠ PROBLEM — check images"
            high_skip_classes.append(class_name)
        elif skip_rate > 20:
            verdict = "~ mild — acceptable"
        else:
            verdict = "✓ good"

        print(f"{class_name:<10} {detected:>10} {skipped:>10} {skip_rate:>9.1f}%  {verdict}")

    print("\n── Summary ──────────────────────────────────────────────────")
    if expected_skip_classes:
        print(f"Non-hand classes (expected high skips): {expected_skip_classes}")
        print("  → These are fine. Consider whether you need them in training.")
    if high_skip_classes:
        print(f"\nProblematic classes with >40% skip: {high_skip_classes}")
        print("  → Open a few images from these classes and check:")
        print("     1. Is the hand clearly visible and well-lit?")
        print("     2. Is the hand severely cropped at the frame edge?")
        print("     3. Is the image very small or blurry?")
        print("  → If images look fine, try lowering min_hand_detection_confidence to 0.1")
    if not high_skip_classes and not expected_skip_classes:
        print("All classes look healthy. Proceed to training.")


if __name__ == "__main__":
    analyze(DATASET_DIR, LANDMARKER_FILE, SAMPLE_PER_CLASS)
