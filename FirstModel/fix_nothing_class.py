"""
FIX NOTHING CLASS
==================
The 'nothing' class has only 14 samples because MediaPipe correctly
finds no hand in those background images and skips them.

Why synthetic zero vectors are the right solution:
    'Nothing' means no hand is present. In our normalized landmark space,
    the correct representation of "no hand" is simply the absence of any
    landmark signal. A zero vector (63 zeros) is a meaningful and honest
    representation of this state — it's not fabricating data, it's encoding
    the semantic meaning of the class directly.

    This is different from other classes where we would never fabricate
    landmarks, because for those classes the actual hand geometry matters.
    For 'nothing', the geometry IS zero by definition.

Why not just rely on the confidence threshold:
    The threshold approach works but is a heuristic — it assumes the model
    will always be uncertain when no hand is present. With a proper nothing
    class, the model explicitly learns to output high confidence for "nothing"
    when the input looks like a zero/noise vector, which is more robust.

Usage:
    python fix_nothing_class.py
    Then re-run: python step2_train_mlp.py
"""

import pandas as pd
import numpy as np

CSV_PATH       = "landmarks.csv"
TARGET_SAMPLES = 2500   # bring nothing up to roughly average class size
NOISE_STD      = 0.01   # tiny noise so samples aren't identical

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Check current state
counts = df["label"].value_counts()
print(f"Current 'nothing' samples: {counts.get('nothing', 0)}")
print(f"Average other class size:  {int(counts[counts.index != 'nothing'].mean())}")

# Remove the 14 existing nothing samples (too few to be reliable)
df = df[df["label"] != "nothing"]

# Generate synthetic nothing samples
# Why small gaussian noise instead of pure zeros:
#   Pure zeros would be identical rows, which can cause numerical issues
#   in BatchNormalization. Tiny noise (std=0.01, much smaller than real
#   landmark variance which is ~0.1-0.5) makes each sample unique while
#   still being clearly distinct from any real hand configuration.
np.random.seed(42)
n_features    = 63
noise         = np.random.normal(0, NOISE_STD, size=(TARGET_SAMPLES, n_features))
feature_cols  = [f"f{i}" for i in range(n_features)]

nothing_df    = pd.DataFrame(noise, columns=feature_cols)
nothing_df.insert(0, "label", "nothing")

# Merge
df_fixed = pd.concat([df, nothing_df], ignore_index=True)

# Shuffle so nothing rows aren't all at the end
df_fixed = df_fixed.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_fixed.to_csv(CSV_PATH, index=False)

print(f"\n✓ Fixed. New class distribution:")
new_counts = df_fixed["label"].value_counts().sort_index()
for label, count in new_counts.items():
    bar     = "█" * (count // 100)
    flag    = " ← synthetic" if label == "nothing" else ""
    print(f"  {label:<10} {count:>5}  {bar}{flag}")

print(f"\n✓ Saved to {CSV_PATH}")
print(f"  Total samples: {len(df_fixed)}")
print("\nNow re-run: python step2_train_mlp.py")
