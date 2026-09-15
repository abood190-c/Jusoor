"""
MIRROR AUGMENTATION — ADD RIGHT HAND SUPPORT
==============================================
The model was trained only on left-hand landmark data from the dataset.
Right-hand landmarks are the horizontal mirror image of left-hand ones.

Why negating X coordinates works:
    MediaPipe landmarks are in normalized coordinates where X goes from
    0 (left edge) to 1 (right edge). After our wrist-origin normalization,
    X values represent horizontal displacement from the wrist.
    Negating X flips the hand left↔right while keeping all finger geometry
    (curl, spread, relative positions) completely intact.
    This is geometrically equivalent to holding up your other hand.

Why we augment the CSV instead of re-extracting:
    Re-extraction would require a mirrored image dataset we don't have.
    Operating directly in landmark space is cleaner, faster, and more
    principled — we're not synthesizing fake images, we're applying a
    geometrically valid transformation to existing landmark data.

What this does:
    For every row in landmarks.csv, creates a mirrored copy by negating
    the x component of all 21 landmarks (every 3rd value starting at f0).
    The original rows stay untouched. Dataset size roughly doubles.

Usage:
    python mirror_augmentation.py
    Then re-run: python step2_train_mlp.py
"""

import pandas as pd
import numpy as np

CSV_PATH = "landmarks.csv"

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"  Original rows: {len(df)}")

feature_cols = [f"f{i}" for i in range(63)]

# ── Identify X feature indices ────────────────────────────────────────
# Landmarks are stored as flattened [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
# So X coordinates are at indices 0, 3, 6, 9, ... (every 3rd starting at 0)
x_indices = list(range(0, 63, 3))   # [0, 3, 6, ..., 60]

print(f"  X coordinate feature indices: {x_indices[:5]}... ({len(x_indices)} total)")

# ── Create mirrored copy ───────────────────────────────────────────────
df_mirror = df.copy()

# Negate all X coordinates — this is the mirror transform
x_col_names = [f"f{i}" for i in x_indices]
df_mirror[x_col_names] = df_mirror[x_col_names] * -1

# ── Skip mirroring for nothing class ──────────────────────────────────
# The nothing class is synthetic zero vectors — mirroring zeros gives zeros,
# so it would just create identical duplicates. We exclude it to avoid
# inflating that class and wasting compute.
df_mirror = df_mirror[df_mirror["label"] != "nothing"]
print(f"  Mirrored rows (excluding nothing): {len(df_mirror)}")

# ── Merge original + mirrored ─────────────────────────────────────────
df_augmented = pd.concat([df, df_mirror], ignore_index=True)

# Shuffle
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────
df_augmented.to_csv(CSV_PATH, index=False)

print(f"\n✓ Augmented dataset saved to {CSV_PATH}")
print(f"  Total rows: {len(df_augmented)}")

# ── Class distribution check ──────────────────────────────────────────
print(f"\nClass distribution after mirroring:")
counts = df_augmented["label"].value_counts().sort_index()
for label, count in counts.items():
    bar  = "█" * (count // 200)
    flag = " ← unchanged (synthetic)" if label == "nothing" else ""
    print(f"  {label:<10} {count:>6}  {bar}{flag}")

print("\nNow re-run: python step2_train_mlp.py")
